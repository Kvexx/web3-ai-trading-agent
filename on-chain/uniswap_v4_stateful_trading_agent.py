"""
Uniswap V4 Stateful Trading Agent
--------------------------------
Enhanced trading agent that maintains state and context between LLM decisions
"""

import os
import sys
import time
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import deque
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box
import traceback

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

# Add the MLX-LM path to Python path
mlx_lm_path = os.path.join(root_dir, 'off-chain', 'rl_trading', 'mlx-lm')
sys.path.insert(0, mlx_lm_path)

from uniswap_v4_stateless_trading_agent import UniswapV4TradingAgent
from config import (
    OLLAMA_MODEL, 
    MODEL_KEY,
    MIN_STRATEGY_DURATION, 
    MAX_STRATEGY_DURATION, 
    get_context_capacity, 
    CONTEXT_WARNING_THRESHOLD, 
    SUMMARIZATION_COOLDOWN,
    DEFAULT_ETH_ALLOCATION,
    USE_MLX_MODEL,
    MLX_BASE_MODEL,
    MLX_ADAPTER_PATH,
    AVAILABLE_MODELS,
    OPENROUTER_API_KEY,
    USE_FORK
)

# Create a logger specifically for the stateful agent
logger = logging.getLogger(__name__)

# Import MLX modules if we're using the custom model
if USE_MLX_MODEL:
    try:
        import mlx.core as mx
        from mlx_lm.utils import load, generate_step, load
        from mlx_lm.sample_utils import make_sampler
        
        # Import the generate function
        try:
            from mlx_lm.generate import generate
        except ImportError:
            try:
                # Fallback to a simplified generate function
                def generate(model, tokenizer, prompt, temp=0.6, max_tokens=500):
                    """Simplified generate function for compatibility"""
                    tokens = tokenizer.encode(prompt)
                    response_tokens = generate_step(model, tokens, temp=temp, max_tokens=max_tokens)
                    return [{"generation": tokenizer.decode(response_tokens)}]
            except Exception as e:
                logger.warning(f"Could not create fallback generate function: {str(e)}")
        
        adapter_path = os.path.abspath(MLX_ADAPTER_PATH) if isinstance(MLX_ADAPTER_PATH, str) else MLX_ADAPTER_PATH
        logger.info(f"MLX modules loaded, will use custom-trained model from {adapter_path}")
    except ImportError as e:
        logger.error(f"Failed to import MLX modules: {str(e)}")
        logger.error("Make sure MLX is installed and in your Python path.")
        logger.error("Falling back to Ollama model.")
        USE_MLX_MODEL = False

@dataclass
class MarketState:
    """Represents the market state at a point in time"""
    timestamp: int
    eth_price: float
    price_change_10m: float
    volume_eth_10m: float
    volume_usdc_10m: float
    swap_count_10m: int

@dataclass
class TradingDecision:
    """Represents a trading decision and its outcome"""
    timestamp: int
    decision_type: str  # "NO_TRADE", "REBALANCE", "LLM_DECISION"
    trade_type: Optional[str]  # "ETH_TO_USDC", "USDC_TO_ETH", None
    amount: float
    reasoning: str
    market_state: MarketState
    portfolio_before: Dict
    portfolio_after: Optional[Dict] = None
    success: Optional[bool] = None
    profit_loss: Optional[float] = None

class TradingContext:
    """Maintains context between trading decisions"""
    def __init__(self, max_history: int = 100):
        self.market_states: deque = deque(maxlen=max_history)
        self.trading_decisions: deque = deque(maxlen=max_history)
        self.performance_metrics: Dict[str, List[float]] = {
            "rebalance_pl": [],
            "llm_pl": [],
        }
        self.current_strategy: Optional[str] = None
        self.strategy_duration: Optional[int] = None
        self.strategy_start_time: Optional[int] = None
        
    def add_market_state(self, state: MarketState):
        self.market_states.append(state)
    
    def add_trading_decision(self, decision: TradingDecision):
        self.trading_decisions.append(decision)
        
        # Update performance metrics if we have profit/loss data
        if decision.profit_loss is not None:
            if decision.decision_type == "REBALANCE":
                self.performance_metrics["rebalance_pl"].append(decision.profit_loss)
            elif decision.decision_type == "LLM_DECISION":
                self.performance_metrics["llm_pl"].append(decision.profit_loss)
    
    def get_recent_decisions(self, minutes: int = 60) -> List[TradingDecision]:
        """Get trading decisions from the last n minutes"""
        cutoff_time = int(time.time()) - (minutes * 60)
        return [d for d in self.trading_decisions if d.timestamp > cutoff_time]
    
    def get_market_trend(self, minutes: int = 60) -> Dict[str, float]:
        """Calculate market trends over the specified time period"""
        states = [s for s in self.market_states 
                 if s.timestamp > int(time.time()) - (minutes * 60)]
        
        if not states:
            return {
                "price_trend": 0.0,
                "volume_trend": 0.0,
                "volatility": 0.0
            }
        
        prices = [s.eth_price for s in states]
        volumes = [s.volume_eth_10m for s in states]
        
        return {
            "price_trend": (prices[-1] - prices[0]) / prices[0] if prices else 0.0,
            "volume_trend": (volumes[-1] - volumes[0]) / volumes[0] if volumes and volumes[0] != 0 else 0.0,
            "volatility": np.std(prices) / np.mean(prices) if prices else 0.0
        }
    
    def get_strategy_performance(self) -> Dict[str, float]:
        """Calculate performance metrics for different strategies"""
        return {
            "rebalance_roi": np.mean(self.performance_metrics["rebalance_pl"]) if self.performance_metrics["rebalance_pl"] else 0.0,
            "llm_roi": np.mean(self.performance_metrics["llm_pl"]) if self.performance_metrics["llm_pl"] else 0.0
        }

class UniswapV4StatefulTradingAgent(UniswapV4TradingAgent):
    """
    Enhanced trading agent that maintains state and context between LLM decisions
    """
    
    def __init__(self, private_key=None, target_eth_allocation=DEFAULT_ETH_ALLOCATION, test_mode=False):
        # Determine model provider and configuration BEFORE calling super().__init__()
        # This ensures model_provider is available when parent class calls _initialize_llm()
        self.model_config = AVAILABLE_MODELS.get(MODEL_KEY.lower())
        if not self.model_config:
            raise ValueError(f"Invalid model key: {MODEL_KEY}. Must be one of: {', '.join(AVAILABLE_MODELS.keys())}")
        
        self.model_provider = self.model_config.get('provider', 'ollama')
        self.model_name = self.model_config['model']
        
        # Context length management from config
        self.context_capacity = get_context_capacity(MODEL_KEY, test_mode)
        self.context_warning_threshold = CONTEXT_WARNING_THRESHOLD
        self.is_summarizing = False
        self.last_summarization_time = 0
        self.summarization_cooldown = SUMMARIZATION_COOLDOWN
        
        # Now call parent init, which will call _initialize_llm()
        super().__init__(private_key, target_eth_allocation)
        self.context = TradingContext()
        
        # Strategy parameters from config
        self.min_strategy_duration = MIN_STRATEGY_DURATION
        self.max_strategy_duration = MAX_STRATEGY_DURATION
        
        # Log model information
        provider_info = f"via {self.model_provider}"
        if USE_MLX_MODEL:
            provider_info = f"via MLX ({MLX_BASE_MODEL})"
        logger.info(f"Using model: {self.model_name} {provider_info} (key: {MODEL_KEY})")
        logger.info(f"Context capacity: {self.context_capacity} tokens (warning threshold: {self.context_warning_threshold:.0%})")
    
    def _initialize_llm(self):
        """Unified method to initialize MLX, Ollama, or OpenRouter LLM"""
        if USE_MLX_MODEL:
            try:
                # Check if adapter files exist
                adapter_path = os.path.abspath(MLX_ADAPTER_PATH) if isinstance(MLX_ADAPTER_PATH, str) else MLX_ADAPTER_PATH
                adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
                adapters_path = os.path.join(adapter_path, "adapters.safetensors")
                
                logger.info(f"Checking adapter directory at: {adapter_path}")
                
                if not os.path.exists(adapter_path):
                    logger.error(f"Adapter directory does not exist: {adapter_path}")
                    raise FileNotFoundError(f"Adapter directory does not exist: {adapter_path}")
                
                if not os.path.exists(adapter_config_path):
                    logger.error(f"adapter_config.json not found at: {adapter_config_path}")
                    raise FileNotFoundError(f"adapter_config.json not found at: {adapter_config_path}")
                    
                if not os.path.exists(adapters_path):
                    logger.error(f"adapters.safetensors not found at: {adapters_path}")
                    raise FileNotFoundError(f"adapters.safetensors not found at: {adapters_path}")
                    
                logger.info(f"Adapter files found, loading model with adapter path: {adapter_path}")
                
                # Use the load function from mlx_lm.utils
                from mlx_lm.utils import load
                self.mlx_model, self.mlx_tokenizer = load(
                    MLX_BASE_MODEL, 
                    adapter_path=adapter_path
                )
                logger.info(f"MLX model loaded successfully with adapter from {adapter_path}")
                return True
            except Exception as e:
                logger.error(f"Error loading MLX model with adapter: {str(e)}")
                # Fall back to base model without adapter
                logger.info(f"Falling back to base model without adapter")
                try:
                    from mlx_lm.utils import load
                    self.mlx_model, self.mlx_tokenizer = load(MLX_BASE_MODEL)
                    logger.info("MLX base model loaded successfully")
                    return True
                except Exception as e:
                    logger.error(f"Error loading MLX base model: {str(e)}")
                    return False
        elif self.model_provider == 'openrouter':
            try:
                # Initialize OpenRouter client
                logger.info(f"Initializing OpenRouter client for {self.model_name}")
                start_time = time.time()
                
                # Import OpenAI client for OpenRouter API compatibility
                try:
                    from openai import OpenAI
                except ImportError:
                    logger.error("OpenAI library not found. Install with: pip install openai")
                    return False
                
                # Initialize OpenRouter client
                self.openrouter_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=OPENROUTER_API_KEY,
                )
                
                logger.info(f"OpenRouter client initialized in {time.time() - start_time:.2f} seconds")
                
                # Test connection with a lightweight request
                logger.info(f"Testing OpenRouter connection...")
                test_start = time.time()
                try:
                    response = self.openrouter_client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a trading assistant."},
                            {"role": "user", "content": "Hello"}
                        ],
                        max_tokens=10,
                        timeout=15
                    )
                    logger.info(f"OpenRouter connection test successful in {time.time() - test_start:.2f} seconds")
                    
                    # Add OpenRouter chat method
                    self._add_openrouter_chat_method()
                    
                    return True
                except Exception as e:
                    logger.error(f"Error testing OpenRouter connection: {e}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error initializing OpenRouter client: {e}")
                return False
        else:
            try:
                # Check if Ollama client is already initialized
                if hasattr(self, 'ollama_client'):
                    logger.info("Ollama client already initialized, skipping initialization")
                    return True
                    
                # Check if OLLAMA_URL is defined in the config, otherwise use a default
                ollama_url = getattr(sys.modules["config"], "OLLAMA_URL", "http://localhost:11434")
                logger.info(f"Initializing Ollama client to connect to {ollama_url}")
                start_time = time.time()
                
                # Initialize client
                from ollama import Client
                self.ollama_client = Client(host=ollama_url)
                logger.info(f"Ollama client initialized in {time.time() - start_time:.2f} seconds")
                
                # Add a helper method for timeout-protected Ollama calls
                self._add_ollama_timeout_handler()
                
                # Test connection with a lightweight request
                logger.info(f"Testing Ollama connection and preloading model...")
                test_start = time.time()
                try:
                    # List available models
                    models = self.ollama_client.list()
                    # Debug the structure of the response
                    try:
                        # Try to serialize response for logging, but handle non-serializable types
                        logger.info(f"Ollama models response received with {len(models.get('models', []))} models")
                    except Exception as e:
                        logger.info(f"Received Ollama models response (not JSON serializable)")
                    
                    # Handle the response structure correctly
                    if 'models' in models:
                        available_models = [model.get('name', model.get('model', '')) for model in models['models']]
                        if OLLAMA_MODEL in available_models:
                            logger.info(f"Ollama connection successful. {OLLAMA_MODEL} is available.")
                        else:
                            logger.warning(f"Model {OLLAMA_MODEL} not found in available models: {available_models}")
                    else:
                        # If models are at the top level
                        available_models = [model.get('name', model.get('model', '')) for model in models.get('models', models)]
                        logger.warning(f"Unexpected Ollama API response structure. Available models: {available_models}")
                    
                    # Send a simple request to preload the model
                    logger.info(f"Preloading {OLLAMA_MODEL} with a simple request")
                    preload_start = time.time()
                    try:
                        # Use our safe chat method with timeout - NO SYSTEM PROMPT for trader-qwen
                        result = self.chat_with_timeout(
                            model=OLLAMA_MODEL, 
                            messages=[
                                {"role": "system", "content": "You are a trading assistant."},
                                {"role": "user", "content": "Hello"}
                            ],
                            timeout=15  # 15 second timeout
                        )
                        logger.info(f"Model preloaded in {time.time() - preload_start:.2f} seconds")
                    except Exception as e:
                        logger.warning(f"Error during model preloading: {e} - continuing anyway")
                        
                    logger.info(f"Ollama initialization completed in {time.time() - test_start:.2f} seconds. Ready for trading.")
                    return True
                except Exception as e:
                    logger.error(f"Error testing Ollama connection: {e}")
                    return False
            except Exception as e:
                logger.error(f"Error initializing Ollama client: {e}")
                logger.info("Will initialize on first request")
                return False

    def _add_ollama_timeout_handler(self):
        """Add a timeout-protected chat method to handle Ollama calls safely"""
        import threading
        import queue
        
        def chat_with_timeout(model, messages, timeout=30):
            """Call Ollama with a timeout to prevent hanging"""
            result_queue = queue.Queue()
            
            def chat_worker():
                try:
                    result = self.ollama_client.chat(
                        model=model,
                        messages=messages
                    )
                    result_queue.put(result)
                except Exception as e:
                    result_queue.put(e)
            
            # Start the chat in a separate thread
            worker_thread = threading.Thread(target=chat_worker)
            worker_thread.daemon = True
            worker_thread.start()
            
            # Wait for result or timeout
            try:
                result = result_queue.get(timeout=timeout)
                if isinstance(result, Exception):
                    raise result
                return result
            except queue.Empty:
                logger.warning(f"Ollama chat timed out after {timeout} seconds")
                # Return a fallback response
                return {
                    "message": {
                        "role": "assistant",
                        "content": "I'm sorry, but I wasn't able to generate a response in time. Please try again or use a different query."
                    }
                }
        
        # Add the method to self
        self.chat_with_timeout = chat_with_timeout

    def _add_openrouter_chat_method(self):
        """Add OpenRouter chat method"""
        def chat_with_openrouter(messages, timeout=30, max_tokens=2000, temperature=0.6):
            """Call OpenRouter with timeout protection"""
            try:
                response = self.openrouter_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout
                )
                return {
                    "message": {
                        "role": "assistant", 
                        "content": response.choices[0].message.content
                    }
                }
            except Exception as e:
                logger.warning(f"OpenRouter chat error: {e}")
                return {
                    "message": {
                        "role": "assistant",
                        "content": "I'm sorry, but I wasn't able to generate a response. Please try again."
                    }
                }
        
        # Add the method to self
        self.chat_with_openrouter = chat_with_openrouter

    def update_context(self, market_data: Dict):
        """Update trading context with new market data"""
        market_state = MarketState(
            timestamp=int(time.time()),
            eth_price=market_data["eth_price"],
            price_change_10m=market_data.get("price_change_pct_10m", 0.0),
            volume_eth_10m=market_data.get("volume_eth_10m", 0.0),
            volume_usdc_10m=market_data.get("volume_usdc_10m", 0.0),
            swap_count_10m=market_data.get("swap_count_10m", 0)
        )
        self.context.add_market_state(market_state)
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count from text"""
        
        # Count words, numbers, and special characters
        word_count = len(text.split())
        # Count digits and special characters that often become separate tokens
        special_char_count = sum(1 for c in text if c.isdigit() or c in '.,;:!?()[]{}"\'+-*/=<>@#$%^&')
        # Count whitespace which often becomes tokens
        whitespace_count = text.count(' ') + text.count('\n') + text.count('\t')
        
        # Use a weighted formula that gives better estimates than simple character division
        estimated_tokens = int(word_count * 1.3 + special_char_count * 0.5 + whitespace_count * 0.5)
        
        # Ensure we don't underestimate
        char_based_estimate = len(text) // 4
        return max(estimated_tokens, char_based_estimate)
    
    def _summarize_and_restart_context(self):
        """Summarize the context and restart with a fresh one, preserving key learning insights."""
        if self.is_summarizing:
            return  # Prevent recursive summarization
        
        current_time = time.time()
        if current_time - self.last_summarization_time < self.summarization_cooldown:
            return  # Respect cooldown period

        self.is_summarizing = True
        try:
            # Get all data from old context
            old_context = self.context
            
            # Extract key learning insights before clearing context
            learning_insights = self._extract_learning_insights(old_context)
            
            # Create a new context with minimal history
            self.context = TradingContext()
            
            # Preserve only the most valuable learning insights (limit to 2 to prevent growth)
            most_valuable_insights = sorted(learning_insights, key=len, reverse=True)[:2]  # Longest insights first
            for insight in most_valuable_insights:
                # Create compact insight decision with shortened reasoning
                compact_insight = insight[:200] + "..." if len(insight) > 200 else insight
                insight_decision = TradingDecision(
                    timestamp=int(time.time()),
                    decision_type="LEARNING_INSIGHT",
                    trade_type=None,
                    amount=0.0,
                    reasoning=compact_insight,
                    market_state=old_context.market_states[-1] if old_context.market_states else None,
                    portfolio_before={}
                )
                self.context.add_trading_decision(insight_decision)
            
            # Preserve performance metrics (keep only recent ones)
            self.context.performance_metrics = {
                "rebalance_pl": old_context.performance_metrics["rebalance_pl"][-3:],  # Keep last 3
                "llm_pl": old_context.performance_metrics["llm_pl"][-3:]  # Keep last 3
            }
            
            # Keep current strategy information
            self.context.current_strategy = old_context.current_strategy
            self.context.strategy_duration = old_context.strategy_duration
            self.context.strategy_start_time = old_context.strategy_start_time
            
            # Keep only the most recent market states (last 2)
            if old_context.market_states:
                for state in list(old_context.market_states)[-2:]:
                    self.context.add_market_state(state)
            
            # Keep only the most essential trades (reduce to prevent context explosion)
            successful_trades = [d for d in old_context.trading_decisions if d.success and d.profit_loss and d.profit_loss > 0]
            recent_trades = [d for d in old_context.trading_decisions if d.trade_type][-2:]  # Last 2 trades only
            
            # Add top 1 most profitable trade only
            if successful_trades:
                top_trade = max(successful_trades, key=lambda x: x.profit_loss or 0)
                self.context.add_trading_decision(top_trade)
            
            # Add most recent trade if it's different from the top trade
            if recent_trades:
                latest_trade = recent_trades[-1]
                existing_trades = [d for d in self.context.trading_decisions if d.trade_type]
                if latest_trade not in existing_trades:
                    self.context.add_trading_decision(latest_trade)
            
            # Log detailed summary information
            total_trades = len([d for d in old_context.trading_decisions if d.trade_type])
            profitable_trades = len(successful_trades)
            logger.info(f"Context summarized with learning insights preserved:\n" +
                         f"- Total trades analyzed: {total_trades}\n" +
                         f"- Profitable trades: {profitable_trades}\n" +
                         f"- Learning insights preserved: {len(learning_insights)}\n" +
                         f"- Key trades retained: {len([d for d in self.context.trading_decisions if d.trade_type])}")
            
            self.last_summarization_time = current_time
        finally:
            self.is_summarizing = False
    
    def _extract_learning_insights(self, old_context: TradingContext) -> List[str]:
        """Extract key learning insights from the context before summarization"""
        insights = []
        
        # Analyze all trades for patterns
        all_trades = [d for d in old_context.trading_decisions if d.trade_type and d.success is not None]
        successful_trades = [d for d in all_trades if d.profit_loss and d.profit_loss > 0]
        failed_trades = [d for d in all_trades if d.profit_loss and d.profit_loss <= 0]
        
        if not all_trades:
            return ["No trading history available for learning insights."]
        
        # Success rate insight
        success_rate = len(successful_trades) / len(all_trades)
        insights.append(f"INSIGHT: Historical success rate is {success_rate:.1%} ({len(successful_trades)}/{len(all_trades)} trades)")
        
        # Best performing trade types
        if successful_trades:
            eth_sells = [d for d in successful_trades if d.trade_type == "ETH_TO_USDC"]
            eth_buys = [d for d in successful_trades if d.trade_type == "USDC_TO_ETH"]
            
            if eth_sells:
                avg_sell_profit = np.mean([d.profit_loss for d in eth_sells])
                insights.append(f"INSIGHT: ETH selling trades averaged {avg_sell_profit:+.1%} profit ({len(eth_sells)} trades)")
            
            if eth_buys:
                avg_buy_profit = np.mean([d.profit_loss for d in eth_buys])
                insights.append(f"INSIGHT: ETH buying trades averaged {avg_buy_profit:+.1%} profit ({len(eth_buys)} trades)")
        
        # Market condition insights
        if successful_trades:
            profitable_prices = [d.market_state.eth_price for d in successful_trades]
            profitable_volatility = [abs(d.market_state.price_change_10m) for d in successful_trades]
            
            avg_profit_price = np.mean(profitable_prices)
            avg_profit_vol = np.mean(profitable_volatility)
            
            insights.append(f"INSIGHT: Most profitable trades occurred around ${avg_profit_price:.0f} ETH price")
            insights.append(f"INSIGHT: Profitable trades averaged {avg_profit_vol:.1f}% volatility")
        
        # Recent trend insight
        recent_trades = all_trades[-5:]  # Last 5 trades
        if recent_trades:
            recent_profits = [d.profit_loss for d in recent_trades if d.profit_loss]
            if recent_profits:
                recent_avg = np.mean(recent_profits)
                trend = "improving" if recent_avg > 0 else "declining"
                insights.append(f"INSIGHT: Recent performance is {trend} (last 5 trades: {recent_avg:+.1%} avg)")
        
        # Risk insight
        if failed_trades:
            avg_loss = np.mean([d.profit_loss for d in failed_trades])
            loss_prices = [d.market_state.eth_price for d in failed_trades]
            avg_loss_price = np.mean(loss_prices)
            insights.append(f"INSIGHT: Losing trades averaged {avg_loss:+.1%} loss, typically around ${avg_loss_price:.0f} ETH")
        
        return insights
    
    def calculate_decision_performance(self, decision: TradingDecision) -> float:
        """Calculate the profit/loss from a trading decision"""
        if not decision.portfolio_before or not decision.portfolio_after:
            return 0.0
        
        # Calculate portfolio values
        eth_price_before = decision.market_state.eth_price
        eth_price_after = self.get_market_data()["eth_price"]
        
        value_before = (decision.portfolio_before["ETH"] * eth_price_before + 
                       decision.portfolio_before["USDC"])
        value_after = (decision.portfolio_after["ETH"] * eth_price_after + 
                      decision.portfolio_after["USDC"])
        
        return (value_after - value_before) / value_before
    
    def make_trading_decision(self) -> Optional[Dict]:
        """Enhanced trading decision logic with state and context"""
        try:
            # First check if we need to rebalance
            should_rebalance, direction, amount = self.should_rebalance()
            
            # Get current market data and update context - with retry logic
            max_retries = 3
            retry_count = 0
            market_data = None
            
            while retry_count < max_retries:
                try:
                    market_data = self.get_market_data()
                    # If successful, update context and break out of retry loop
                    self.update_context(market_data)
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.error(f"Failed to get market data after {max_retries} attempts: {e}")
                        # Create a simple market_data with current price only
                        market_data = {
                            "eth_price": self.swapper.get_eth_price(),
                            "price_change_pct_10m": 0.0,
                            "volume_eth_10m": 0.0,
                            "volume_usdc_10m": 0.0,
                            "swap_count_10m": 0,
                            "timestamp": int(time.time())
                        }
                        # Update context with limited data
                        self.update_context(market_data)
                    else:
                        logger.warning(f"Error getting market data (attempt {retry_count}): {e}, retrying...")
                        time.sleep(1)  # Wait a bit before retrying
            
            if should_rebalance:
                # Create a rebalancing decision
                decision = TradingDecision(
                    timestamp=int(time.time()),
                    decision_type="REBALANCE",
                    trade_type=direction,
                    amount=amount,
                    reasoning="Portfolio rebalancing to maintain target allocation",
                    market_state=self.context.market_states[-1],
                    portfolio_before=self.portfolio.copy()
                )
                
                # Execute the trade
                trade_result = self.execute_trade(
                    trade_type=direction,
                    amount=amount,
                    reason=decision.reasoning
                )
                
                if trade_result:
                    # Update decision with results
                    decision.portfolio_after = self.portfolio.copy()
                    decision.success = trade_result["status"] == "success"
                    decision.profit_loss = self.calculate_decision_performance(decision)
                    
                    # Add to context
                    self.context.add_trading_decision(decision)
                    return trade_result
            
            # If no rebalancing needed, use enhanced LLM decision making
            return self.make_llm_trading_decision()
        except Exception as e:
            logger.error(f"Error in make_trading_decision: {e}")
            logger.error(traceback.format_exc())  # Log full stack trace
            return None
    
    def make_llm_trading_decision(self) -> Optional[Dict]:
        """Enhanced LLM-based trading decision with context and state"""
        # Get current market data
        market_data = self.get_market_data()
        
        # Get market trends
        market_trends = self.context.get_market_trend(minutes=60)
        
        # Get all available trading decisions without time limitation
        recent_decisions = list(self.context.trading_decisions)
        
        # Get strategy performance
        strategy_perf = self.context.get_strategy_performance()
        
        # Check if we need to start a new strategy
        current_time = int(time.time())
        if (not self.context.current_strategy or 
            self.context.strategy_start_time is None or
            self.context.strategy_duration is None or
            current_time - self.context.strategy_start_time > self.context.strategy_duration):
            needs_new_strategy = True
        else:
            needs_new_strategy = False
        
        # Calculate ETH volatility
        volatility = market_trends["volatility"] * 100
        
        # Calculate recent price change in ticks (scaled to match training data format)
        price_change_ticks = market_data.get("price_change_pct_10m", 0.0) * 10
        
        # Build enhanced prompt with historical context and learning
        historical_context = self._build_historical_context()
        performance_insights = self._analyze_trading_performance()
        market_insights = self._analyze_market_patterns()
        
        # Check if we have enough history to include learning context
        # If context usage is already high, use simplified prompt
        if hasattr(self, 'last_context_usage') and self.last_context_usage.get('ratio', 0) > 0.7:
            # Use simplified prompt when context is getting full
            user_query = f"""Given ETH price is ${market_data["eth_price"]:.2f} with volume of {market_data.get("volume_eth_10m", 0.0):.2f} and volatility of {volatility:.4f}, 
recent price change of {price_change_ticks:.4f} ticks, and I currently hold {self.portfolio["ETH"]:.1f} ETH and {self.portfolio["USDC"]:.0f} USDC, 
what trading action should I take on Uniswap?

{historical_context}"""
        else:
            # Use full enhanced prompt when context allows
            user_query = f"""Given ETH price is ${market_data["eth_price"]:.2f} with volume of {market_data.get("volume_eth_10m", 0.0):.2f} and volatility of {volatility:.4f}, 
recent price change of {price_change_ticks:.4f} ticks, and I currently hold {self.portfolio["ETH"]:.1f} ETH and {self.portfolio["USDC"]:.0f} USDC.

TRADING HISTORY & PERFORMANCE:
{historical_context}

PERFORMANCE INSIGHTS:
{performance_insights}

MARKET PATTERN ANALYSIS:
{market_insights}

Based on this historical data and performance analysis, what trading action should I take on Uniswap?"""

        
        # Estimate token count for the prompt
        prompt_token_count = self._estimate_token_count(user_query)
        
        # Estimate token count for all accumulated decisions
        context_token_count = 0
        for decision in self.context.trading_decisions:
            if decision.reasoning:
                context_token_count += self._estimate_token_count(decision.reasoning)
        
        # Add some tokens for each market state
        market_state_tokens = len(self.context.market_states) * 20  # Approximate token count per market state
        
        # Total estimated tokens
        total_estimated_tokens = prompt_token_count + context_token_count + market_state_tokens
        context_usage_ratio = total_estimated_tokens / self.context_capacity
        
        # Store context usage info for later display
        self.last_context_usage = {
            'tokens': total_estimated_tokens,
            'ratio': context_usage_ratio
        }
        
        logger.info(f"Context usage: {total_estimated_tokens} tokens ({context_usage_ratio:.1%}) - History: {len(recent_decisions)} trades, {len(self.context.market_states)} market states")
        
        # Check if we need to restart with summarized context
        if context_usage_ratio > self.context_warning_threshold and not self.is_summarizing:
            logger.warning(f"Context usage approaching limit ({context_usage_ratio:.1%}), summarizing context")
            self._summarize_and_restart_context()
            time.sleep(1)  # Small delay to ensure summarization is complete
        
        try:
            # Get response from either MLX model or Ollama
            if USE_MLX_MODEL:
                # Use the custom-trained MLX model
                system_prompt = "You are a trading assistant that helps users make trading decisions based on market data. You first use chain-of-draft reasoning with short steps, then provide a clear trading action."
                full_prompt = f"{system_prompt}\n\n{user_query}"
                
                # Load the model if not already loaded
                if not hasattr(self, 'mlx_model'):
                    logger.info(f"Loading MLX model {MLX_BASE_MODEL} with adapter {MLX_ADAPTER_PATH}")
                    try:
                        # Check if adapter files exist
                        adapter_path = MLX_ADAPTER_PATH
                        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
                        adapters_path = os.path.join(adapter_path, "adapters.safetensors")
                        
                        logger.info(f"Checking adapter directory at: {adapter_path}")
                        
                        if not os.path.exists(adapter_path):
                            logger.error(f"Adapter directory does not exist: {adapter_path}")
                            raise FileNotFoundError(f"Adapter directory does not exist: {adapter_path}")
                        
                        if not os.path.exists(adapter_config_path):
                            logger.error(f"adapter_config.json not found at: {adapter_config_path}")
                            raise FileNotFoundError(f"adapter_config.json not found at: {adapter_config_path}")
                            
                        if not os.path.exists(adapters_path):
                            logger.error(f"adapters.safetensors not found at: {adapters_path}")
                            raise FileNotFoundError(f"adapters.safetensors not found at: {adapters_path}")
                            
                        logger.info(f"Adapter files found, loading model with adapter path: {adapter_path}")
                        
                        self.mlx_model, self.mlx_tokenizer = load(
                            MLX_BASE_MODEL, 
                            adapter_path=adapter_path
                        )
                    except Exception as e:
                        logger.error(f"Error loading MLX model with adapter: {str(e)}")
                        # Fall back to base model without adapter
                        logger.info(f"Falling back to base model without adapter")
                        self.mlx_model, self.mlx_tokenizer = load(MLX_BASE_MODEL)
                
                # Generate response
                tokens = self.mlx_tokenizer.encode(full_prompt)
                results = generate(
                    self.mlx_model,
                    self.mlx_tokenizer,
                    prompt=full_prompt,
                    max_tokens=500,
                    sampler=make_sampler(temp=0.6)
                )
                # Handle different return types from generate function
                if isinstance(results, list) and isinstance(results[0], dict):
                    response_text = results[0]["generation"]
                elif isinstance(results, str):
                    response_text = results
                else:
                    response_text = str(results)
                logger.info(f"MLX Model Response: {response_text}")
            elif self.model_provider == 'openrouter':
                # Call OpenRouter API
                start_time = time.time()
                try:
                    response = self.chat_with_openrouter(
                        messages=[
                            {"role": "system", "content": "You are a trading assistant that helps users make trading decisions based on market data. You first use chain-of-draft reasoning with short steps, then provide a clear trading action."},
                            {"role": "user", "content": user_query}
                        ],
                        timeout=90,  # 90 second timeout for trading decisions
                        max_tokens=2000,  # 2000 for OpenRouter Grok-4; 500 is suggested for fine-tuned custom chain-of-draft models
                        temperature=0.6
                    )
                    
                    response_text = response.get("message", {}).get("content", "")
                    
                    # Note: Don't truncate OpenRouter responses - they don't use RECOMMENDATION format
                    logger.info(f"OpenRouter response received in {time.time() - start_time:.2f} seconds")
                except Exception as e:
                    logger.error(f"Error getting OpenRouter response: {e}")
                    response_text = "ERROR: Model response timed out"
            else:
                # Call Ollama with timeout protection for regular trading decisions
                start_time = time.time()
                try:
                    # Use our safe chat method with timeout - NO SYSTEM PROMPT for trader-qwen
                    response = self.chat_with_timeout(
                        model=OLLAMA_MODEL,
                        messages=[
                            {
                                "role": "user",
                                "content": user_query
                            }
                        ],
                        timeout=90  # 90 second timeout for trading decisions
                    )
                    
                    response_text = response.get("message", {}).get("content", "")
                    
                    # Truncate response after RECOMMENDATION to avoid garbage content (Ollama models only)
                    recommendation_match = re.search(r"RECOMMENDATION:.*$", response_text, re.MULTILINE)
                    if recommendation_match:
                        # Keep only up to the end of the recommendation line
                        end_pos = recommendation_match.end()
                        response_text = response_text[:end_pos]
                        
                    logger.info(f"Observation response received in {time.time() - start_time:.2f} seconds")
                except Exception as e:
                    logger.error(f"Error getting observation response: {e}")
                    response_text = "ERROR: Model response timed out"
            
            logger.info(f"\n--- OBSERVATION #{len(self.context.trading_decisions)+1} ---\n{response_text}\n----------------------------")
            
            # Print response to console so user can see it
            print(f"\n--- OBSERVATION #{len(self.context.trading_decisions)+1} ---")
            print(response_text)
            print("----------------------------")
            
            # Extract trading decision - different parsing for different model types
            if USE_MLX_MODEL or self.model_provider != 'openrouter':
                # Look for RECOMMENDATION pattern (MLX/Ollama format)
                canary_match = re.search(r"RECOMMENDATION:\s+APE\s+(IN|OUT|NEUTRAL)(?:\s+([\d\.]+|part of|some)\s*(?:ETH)?)?", response_text)
            else:
                # Parse natural language responses from OpenRouter models (e.g., Grok-4)
                parsed_decision = self._parse_natural_language_trading_decision(response_text)
                canary_match = parsed_decision
            
            if canary_match:
                action = canary_match.group(1).upper()
                
                # Get amount string if available
                amount_str = canary_match.group(2) if len(canary_match.groups()) >= 2 else None
                current_time = int(time.time())
                
                # Convert action to trade parameters
                if action == "OUT":
                    trade_type = "ETH_TO_USDC"
                    
                    # Determine amount based on the amount string
                    amount = self._parse_trading_amount(amount_str, "ETH_TO_USDC")
                    
                elif action == "IN":
                    trade_type = "USDC_TO_ETH"
                    
                    # Determine amount based on the amount string
                    amount = self._parse_trading_amount(amount_str, "USDC_TO_ETH")
                    
                else:  # NEUTRAL
                    # For NEUTRAL, don't trade
                    decision = TradingDecision(
                        timestamp=current_time,
                        decision_type="LLM_DECISION",
                        trade_type=None,
                        amount=0.0,
                        reasoning=response_text,
                        market_state=self.context.market_states[-1],
                        portfolio_before=self.portfolio.copy()
                    )
                    self.context.add_trading_decision(decision)
                    return
                
                # Add minimum trade size check
                if trade_type == "ETH_TO_USDC" and amount < 0.00001:
                    logger.warning(f"ETH amount {amount:.6f} too small for trade, minimum is 0.00001 ETH")
                    
                    # Record as observation without trade
                    decision = TradingDecision(
                        timestamp=int(time.time()),
                        decision_type="OBSERVATION",
                        trade_type=None,
                        amount=0.0,
                        reasoning=response_text,
                        market_state=self.context.market_states[-1],
                        portfolio_before=self.portfolio.copy()
                    )
                    self.context.add_trading_decision(decision)
                    return
                    
                elif trade_type == "USDC_TO_ETH":
                    market_data = self.get_market_data()
                    if amount * market_data["eth_price"] < 0.01:
                        logger.warning(f"USDC amount {amount * market_data['eth_price']:.6f} too small for trade, minimum is 0.01 USDC")
                        
                        # Record as observation without trade
                        decision = TradingDecision(
                            timestamp=int(time.time()),
                            decision_type="OBSERVATION",
                            trade_type=None,
                            amount=0.0,
                            reasoning=response_text,
                            market_state=self.context.market_states[-1],
                            portfolio_before=self.portfolio.copy()
                        )
                        self.context.add_trading_decision(decision)
                        return
                
                # Create trading decision
                decision = TradingDecision(
                    timestamp=current_time,
                    decision_type="LLM_DECISION",
                    trade_type=trade_type,
                    amount=amount,
                    reasoning=response_text,
                    market_state=self.context.market_states[-1],
                    portfolio_before=self.portfolio.copy()
                )
                
                # Execute trade
                trade_result = self.execute_trade(
                    trade_type=trade_type,
                    amount=amount,
                    reason=response_text
                )
                
                if trade_result:
                    # Update decision with results
                    decision.portfolio_after = self.portfolio.copy()
                    decision.success = trade_result["status"] == "success"
                    decision.profit_loss = self.calculate_decision_performance(decision)
                    
                    # Add to context
                    self.context.add_trading_decision(decision)
                    return
            else:
                # No recommendation found, record as observation only
                decision = TradingDecision(
                    timestamp=int(time.time()),
                    decision_type="OBSERVATION",
                    trade_type=None,
                    amount=0.0,
                    reasoning=response_text,
                    market_state=self.context.market_states[-1],
                    portfolio_before=self.portfolio.copy()
                )
                self.context.add_trading_decision(decision)
            
        except Exception as e:
            logger.error(f"Error in observation analysis: {e}")
            traceback.print_exc()
    
    def _parse_natural_language_trading_decision(self, response_text: str):
        """
        Parse natural language trading recommendations from OpenRouter models like Grok-4.
        Returns an object that mimics the structure expected by the existing trade execution logic.
        """
        try:
            # Convert to lowercase for easier matching
            text_lower = response_text.lower()
            
            # Define trading action patterns
            sell_patterns = [
                r'sell(?:ing)?\s+(?:approximately\s+)?(?:about\s+)?([\d\.]+)(?:\s+|-)?eth',
                r'(?:recommend|suggest)(?:ing)?\s+(?:to\s+)?sell(?:ing)?\s+(?:approximately\s+)?(?:about\s+)?([\d\.]+)(?:\s+|-)?eth',
                r'(?:would|should)\s+sell\s+(?:approximately\s+)?(?:about\s+)?([\d\.]+)(?:\s+|-)?eth',
                r'disposing?\s+of\s+(?:approximately\s+)?(?:about\s+)?([\d\.]+)(?:\s+|-)?eth',
                r'liquidat(?:e|ing)\s+(?:approximately\s+)?(?:about\s+)?([\d\.]+)(?:\s+|-)?eth',
                r'convert(?:ing)?\s+(?:approximately\s+)?(?:about\s+)?([\d\.]+)(?:\s+|-)?eth\s+(?:to|into)\s+usdc',
                r'swap(?:ping)?\s+(?:approximately\s+)?(?:about\s+)?([\d\.]+)(?:\s+|-)?eth\s+(?:to|for)\s+usdc',
                # Percentage-based selling
                r'sell\s+(?:approximately\s+)?(?:about\s+)?([\d\.]+)%\s+of\s+(?:the\s+)?eth',
                r'(?:recommend|suggest)(?:ing)?\s+(?:to\s+)?sell\s+(?:approximately\s+)?(?:about\s+)?([\d\.]+)%',
                # Small amount selling
                r'sell\s+(?:a\s+)?(?:small\s+)?(?:portion|amount|part)\s+(?:of\s+(?:the\s+)?eth)?',
                r'(?:recommend|suggest)(?:ing)?\s+(?:to\s+)?sell\s+(?:a\s+)?(?:small\s+)?(?:portion|amount|part)',
            ]
            
            buy_patterns = [
                r'buy(?:ing)?\s+(?:approximately\s+)?(?:about\s+)?([\d\.]+)(?:\s+|-)?eth',
                r'(?:recommend|suggest)(?:ing)?\s+(?:to\s+)?buy(?:ing)?\s+(?:approximately\s+)?(?:about\s+)?([\d\.]+)(?:\s+|-)?eth',
                r'(?:would|should)\s+buy\s+(?:approximately\s+)?(?:about\s+)?([\d\.]+)(?:\s+|-)?eth',
                r'purchas(?:e|ing)\s+(?:approximately\s+)?(?:about\s+)?([\d\.]+)(?:\s+|-)?eth',
                r'acquir(?:e|ing)\s+(?:approximately\s+)?(?:about\s+)?([\d\.]+)(?:\s+|-)?eth',
                r'convert(?:ing)?\s+(?:approximately\s+)?(?:about\s+)?\$?([\d\.]+)\s+usdc\s+(?:to|into)\s+eth',
                r'swap(?:ping)?\s+(?:approximately\s+)?(?:about\s+)?\$?([\d\.]+)\s+usdc\s+(?:to|for)\s+eth',
                # USDC amount to ETH
                r'spend(?:ing)?\s+(?:approximately\s+)?(?:about\s+)?\$?([\d\.]+)\s+usdc\s+(?:on|for)\s+eth',
                # Percentage-based buying
                r'buy\s+(?:approximately\s+)?(?:about\s+)?([\d\.]+)%\s+(?:worth\s+)?(?:of\s+)?eth',
                r'(?:recommend|suggest)(?:ing)?\s+(?:to\s+)?(?:invest|put)\s+(?:approximately\s+)?(?:about\s+)?([\d\.]+)%\s+(?:in|into)\s+eth',
            ]
            
            hold_patterns = [
                r'hold(?:\s+the)?(?:\s+current)?(?:\s+position)?',
                r'maintain(?:\s+the)?(?:\s+current)?(?:\s+position)?',
                r'(?:do\s+)?(?:not|nothing|no)\s+(?:trade|trading|action)',
                r'stay\s+(?:the\s+course|put)',
                r'keep\s+(?:the\s+)?(?:current\s+)?(?:position|allocation)',
                r'remain\s+(?:in\s+)?(?:the\s+)?(?:current\s+)?position',
                r'neutral',
                r'wait(?:\s+and\s+see)?',
                r'observe(?:\s+the\s+market)?',
                r'no\s+(?:immediate\s+)?action\s+(?:needed|required)',
            ]
            
            # Check for sell signals
            for pattern in sell_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    amount_str = match.group(1) if match.groups() else None
                    logger.info(f"Detected SELL signal with amount: {amount_str}")
                    return self._create_mock_match("OUT", amount_str)
            
            # Check for buy signals  
            for pattern in buy_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    amount_str = match.group(1) if match.groups() else None
                    logger.info(f"Detected BUY signal with amount: {amount_str}")
                    return self._create_mock_match("IN", amount_str)
            
            # Check for hold/neutral signals
            for pattern in hold_patterns:
                if re.search(pattern, text_lower):
                    logger.info(f"Detected HOLD/NEUTRAL signal")
                    return self._create_mock_match("NEUTRAL", None)
            
            # If no clear pattern is found, log and return None
            logger.info(f"No clear trading signal detected in response")
            return None
            
        except Exception as e:
            logger.error(f"Error parsing natural language trading decision: {e}")
            return None
    
    def _create_mock_match(self, action: str, amount_str: str):
        """
        Create a mock match object that mimics the structure expected by existing code.
        This allows us to reuse the existing trade execution logic.
        """
        class MockMatch:
            def __init__(self, action, amount_str):
                self._groups = [action]
                if amount_str is not None:
                    self._groups.append(amount_str)
            
            def group(self, index):
                if index == 1:
                    return self._groups[0]  # action (IN/OUT/NEUTRAL)
                elif index == 2 and len(self._groups) > 1:
                    return self._groups[1]  # amount
                return None
            
            def groups(self):
                return tuple(self._groups[1:])  # Return everything except action as groups
        
        return MockMatch(action, amount_str)
    
    def _parse_trading_amount(self, amount_str: str, trade_type: str) -> float:
        """
        Parse trading amount from natural language text, handling various formats.
        Returns the amount in ETH regardless of trade direction.
        """
        try:
            if not amount_str:
                # Default amount if not specified (conservative 2%)
                if trade_type == "ETH_TO_USDC":
                    return 0.02 * self.portfolio["ETH"]
                else:  # USDC_TO_ETH
                    market_data = self.get_market_data()
                    usdc_amount = 0.02 * self.portfolio["USDC"]
                    return usdc_amount / market_data["eth_price"]
            
            # Clean the amount string
            amount_clean = amount_str.strip().lower()
            
            # Handle percentage amounts (e.g., "5%", "10%")
            if '%' in amount_clean:
                percentage = float(amount_clean.replace('%', ''))
                percentage_decimal = percentage / 100.0
                
                if trade_type == "ETH_TO_USDC":
                    return percentage_decimal * self.portfolio["ETH"]
                else:  # USDC_TO_ETH
                    market_data = self.get_market_data()
                    usdc_amount = percentage_decimal * self.portfolio["USDC"]
                    return usdc_amount / market_data["eth_price"]
            
            # Handle direct numeric amounts (e.g., "1.5", "0.1")
            if amount_clean.replace('.', '', 1).isdigit():
                numeric_amount = float(amount_clean)
                
                # For ETH_TO_USDC, amount is already in ETH
                if trade_type == "ETH_TO_USDC":
                    return numeric_amount
                else:  # USDC_TO_ETH
                    # Check if this looks like a USD amount (> 100) or ETH amount
                    if numeric_amount > 100:
                        # Treat as USDC amount, convert to ETH
                        market_data = self.get_market_data()
                        return numeric_amount / market_data["eth_price"]
                    else:
                        # Treat as ETH amount
                        return numeric_amount
            
            # Handle descriptive amounts
            descriptive_amounts = {
                'small': 0.01,      # 1%
                'little': 0.01,     # 1%
                'tiny': 0.005,      # 0.5%
                'portion': 0.02,    # 2%
                'part': 0.02,       # 2%
                'some': 0.05,       # 5%
                'moderate': 0.1,    # 10%
                'significant': 0.2, # 20%
                'large': 0.3,       # 30%
                'substantial': 0.4, # 40%
                'major': 0.5,       # 50%
            }
            
            for word, percentage in descriptive_amounts.items():
                if word in amount_clean:
                    if trade_type == "ETH_TO_USDC":
                        return percentage * self.portfolio["ETH"]
                    else:  # USDC_TO_ETH
                        market_data = self.get_market_data()
                        usdc_amount = percentage * self.portfolio["USDC"]
                        return usdc_amount / market_data["eth_price"]
            
            # Default fallback (conservative 2%)
            if trade_type == "ETH_TO_USDC":
                return 0.02 * self.portfolio["ETH"]
            else:  # USDC_TO_ETH
                market_data = self.get_market_data()
                usdc_amount = 0.02 * self.portfolio["USDC"]
                return usdc_amount / market_data["eth_price"]
                
        except Exception as e:
            logger.warning(f"Error parsing trading amount '{amount_str}': {e}")
            # Safe fallback amount (1% of holdings)
            if trade_type == "ETH_TO_USDC":
                return 0.01 * self.portfolio["ETH"]
            else:  # USDC_TO_ETH
                market_data = self.get_market_data()
                usdc_amount = 0.01 * self.portfolio["USDC"]
                return usdc_amount / market_data["eth_price"]
    
    def _format_recent_decisions(self, decisions: List[TradingDecision]) -> str:
        """Format recent trading decisions for the prompt"""
        if not decisions:
            return "No recent trades"
        
        formatted = []
        for d in decisions:
            if d.trade_type:
                result = "✓" if d.success else "✗"
                pl = f" ({d.profit_loss:+.2%})" if d.profit_loss is not None else ""
                formatted.append(
                    f"[{datetime.fromtimestamp(d.timestamp).strftime('%H:%M:%S')}] "
                    f"{d.decision_type}: {d.trade_type} {d.amount:.4f} ETH {result}{pl}"
                )
            else:
                formatted.append(
                    f"[{datetime.fromtimestamp(d.timestamp).strftime('%H:%M:%S')}] "
                    f"NO TRADE"
                )
        
        # Use all available decisions without limit
        return "\n".join(formatted)
    
    def _build_historical_context(self) -> str:
        """Build concise historical context showing recent trades and their outcomes"""
        # Get learning insights from previous context summarizations (limit to prevent growth)
        learning_insights = [d for d in self.context.trading_decisions 
                           if d.decision_type == "LEARNING_INSIGHT"][-2:]  # Only last 2 insights
        
        # Get actual recent trades (reduced from 10 to 5 for context efficiency)
        recent_trades = [d for d in self.context.trading_decisions 
                        if d.trade_type and d.success is not None][-5:]  # Last 5 trades
        
        context_lines = []
        
        # Include learning insights first (condensed format)
        if learning_insights:
            context_lines.append("Key insights:")
            for insight in learning_insights:
                # Extract just the core insight, not the full reasoning
                core_insight = insight.reasoning.replace("INSIGHT: ", "").split('.')[0]
                context_lines.append(f"- {core_insight}")
            context_lines.append("")  # Empty line separator
        
        # Include recent trades (condensed format)
        if recent_trades:
            context_lines.append(f"Last {len(recent_trades)} trades:")
            for trade in recent_trades:
                result = "+" if trade.profit_loss and trade.profit_loss > 0 else "-"
                pl_pct = f"{trade.profit_loss:+.1%}" if trade.profit_loss else "0%"
                direction = "SELL" if trade.trade_type == "ETH_TO_USDC" else "BUY"
                
                context_lines.append(f"- {direction} {result}{pl_pct}")
        else:
            if not learning_insights:
                return "No previous trades."
        
        return "\n".join(context_lines)
    
    def _analyze_trading_performance(self) -> str:
        """Analyze trading performance to identify successful patterns (concise version)"""
        successful_trades = [d for d in self.context.trading_decisions 
                           if d.trade_type and d.success and d.profit_loss and d.profit_loss > 0]
        failed_trades = [d for d in self.context.trading_decisions 
                        if d.trade_type and d.success and d.profit_loss and d.profit_loss <= 0]
        
        if not successful_trades and not failed_trades:
            return "No performance data."
        
        insights = []
        
        # Success rate analysis (compact)
        total_trades = len(successful_trades) + len(failed_trades)
        success_rate = len(successful_trades) / total_trades if total_trades > 0 else 0
        insights.append(f"Success: {success_rate:.0%} ({len(successful_trades)}/{total_trades})")
        
        # Trade type performance (compact)
        eth_sells = [d.profit_loss for d in successful_trades if d.trade_type == "ETH_TO_USDC"]
        eth_buys = [d.profit_loss for d in successful_trades if d.trade_type == "USDC_TO_ETH"]
        
        if eth_sells:
            insights.append(f"ETH sells: {len(eth_sells)} trades, {np.mean(eth_sells):+.1%} avg")
        
        if eth_buys:
            insights.append(f"ETH buys: {len(eth_buys)} trades, {np.mean(eth_buys):+.1%} avg")
        
        return "\n".join(insights)
    
    def _analyze_market_patterns(self) -> str:
        """Analyze market conditions where trades were successful vs unsuccessful (concise version)"""
        successful_trades = [d for d in self.context.trading_decisions 
                           if d.trade_type and d.success and d.profit_loss and d.profit_loss > 0]
        
        if len(successful_trades) < 2:
            return "Insufficient pattern data."
        
        insights = []
        
        # Price level analysis (compact)
        successful_prices = [d.market_state.eth_price for d in successful_trades]
        avg_price = np.mean(successful_prices)
        insights.append(f"Profitable at ~${avg_price:.0f}")
        
        # Market timing insights (compact)
        current_market = self.context.market_states[-1] if self.context.market_states else None
        if current_market:
            similar_conditions = []
            for trade in successful_trades:
                price_diff = abs(trade.market_state.eth_price - current_market.eth_price) / current_market.eth_price
                if price_diff < 0.05:  # Similar price (±5%)
                    similar_conditions.append(trade)
            
            if similar_conditions:
                avg_profit = np.mean([d.profit_loss for d in similar_conditions if d.profit_loss])
                insights.append(f"Similar conditions: {len(similar_conditions)} trades, {avg_profit:+.1%} avg")
        
        return "\n".join(insights) if insights else "No patterns yet."
    
    def display_portfolio(self):
        """Enhanced portfolio display with strategy information"""
        super().display_portfolio()
        
        # Add strategy information
        if self.context.current_strategy:
            strategy_table = Table(title="Current Trading Strategy", box=box.MINIMAL_HEAVY_HEAD)
            strategy_table.add_column("Metric", style="cyan")
            strategy_table.add_column("Value", justify="right")
            
            strategy_table.add_row("Strategy", self.context.current_strategy)
            if self.context.strategy_start_time:
                elapsed = str(timedelta(seconds=int(time.time() - self.context.strategy_start_time)))
                remaining = str(timedelta(seconds=max(0, self.context.strategy_duration - 
                                                   (int(time.time()) - self.context.strategy_start_time))))
                strategy_table.add_row("Elapsed Time", elapsed)
                strategy_table.add_row("Remaining Time", remaining)
            
            # Add performance metrics
            perf = self.context.get_strategy_performance()
            strategy_table.add_row("Rebalancing ROI", f"{perf['rebalance_roi']:.2%}")
            strategy_table.add_row("LLM Trading ROI", f"{perf['llm_roi']:.2%}")
            
            console = Console()
            console.print(strategy_table)

    def run(self, iterations=None):
        """Run the trading loop"""
        iteration = 0
        
        while True:
            if iterations is not None and iteration >= iterations:
                break
            
            print(f"\nMaking trading decision at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Make trading decision
            self.make_trading_decision()
            
            # Display portfolio and market data (this also displays trade history)
            self.display_portfolio()
            
            # Display context usage
            if hasattr(self, 'last_context_usage'):
                logger.info(f"Estimated context usage: {self.last_context_usage['tokens']:.0f} tokens ({self.last_context_usage['ratio']:.1%})")
            
            print("Waiting 10 seconds before next decision...")
            time.sleep(10)
            
            iteration += 1

    def run_observation_mode(self, target_tokens=None, target_minutes=None, target_cycles=None):
        """Run the trading loop in observation mode"""
        logger.info("Starting observation mode - collecting market data without executing trades")
        start_time = time.time()
        cycle_count = 0
        
        while True:
            # Get market data and update context
            market_data = self.get_market_data()
            self.update_context(market_data)
            cycle_count += 1
            
            # Process decision but don't execute trade
            if hasattr(self, 'ollama_client') or hasattr(self, 'openrouter_client') or USE_MLX_MODEL:
                # Run LLM decision making without executing trades
                self.make_llm_trading_decision_observe_only()
            
            # Display portfolio and market data
            self.display_portfolio()
            print(f"OBSERVATION MODE: Collected {cycle_count} observation cycles")
            
            # Display context usage
            if hasattr(self, 'last_context_usage'):
                logger.info(f"Estimated context usage: {self.last_context_usage['tokens']:.0f} tokens ({self.last_context_usage['ratio']:.1%})")
            
            # Check if we need to stop observation mode
            if target_tokens and hasattr(self, 'last_context_usage') and self.last_context_usage['tokens'] >= target_tokens:
                logger.info(f"Observation mode complete: Target {target_tokens} tokens reached")
                break
            elif target_minutes and (time.time() - start_time) / 60 >= target_minutes:
                logger.info(f"Observation mode complete: Target {target_minutes} minutes reached")
                break
            elif target_cycles and cycle_count >= target_cycles:
                logger.info(f"Observation mode complete: Target {target_cycles} cycles reached")
                break
            
            print("Waiting 10 seconds before next observation...")
            time.sleep(10)
        
        # Log comprehensive summary of observations
        observation_count = sum(1 for d in self.context.trading_decisions if d.decision_type == "OBSERVATION")
        unique_prices = set(round(s.eth_price, 2) for s in self.context.market_states)
        price_range = (min(s.eth_price for s in self.context.market_states), 
                      max(s.eth_price for s in self.context.market_states)) if self.context.market_states else (0, 0)
        
        logger.info(f"Observation phase complete - collected data summary:")
        logger.info(f"- Total observations: {observation_count}")
        logger.info(f"- Observation period: {timedelta(seconds=int(time.time() - start_time))}")
        logger.info(f"- Price range: ${price_range[0]:.2f} - ${price_range[1]:.2f} (${price_range[1]-price_range[0]:.2f} spread)")
        logger.info(f"- Unique price points: {len(unique_prices)}")
        
        # Instead of summarizing/restarting context, we want to preserve ALL observations
        # for the initial strategy generation, so we'll skip that step
        # self._summarize_and_restart_context()
        
        # Generate initial strategy based on observations
        self._generate_initial_strategy()
        
        # After generating the strategy, we can summarize the context to save space
        # while preserving the generated strategy and key insights
        self._summarize_and_restart_context()
        
        logger.info("Observation mode complete - switching to active trading")
    
    def make_llm_trading_decision_observe_only(self):
        """Simulated LLM decision making for observation mode - no actual trades"""
        # Get current market data
        market_data = self.get_market_data()
        
        # Get market trends
        market_trends = self.context.get_market_trend(minutes=60)
        
        # Get all available trading decisions without time limitation
        recent_decisions = list(self.context.trading_decisions)
        
        # Calculate ETH volatility
        volatility = market_trends["volatility"] * 100
        
        # Calculate recent price change in ticks (scaled to match training data format)
        price_change_ticks = market_data.get("price_change_pct_10m", 0.0) * 10
        
        # Build enhanced prompt with historical context and learning (for observation mode)
        historical_context = self._build_historical_context()
        performance_insights = self._analyze_trading_performance()
        market_insights = self._analyze_market_patterns()
        
        # Prepare enhanced user query with learning context
        user_query = f"""Given ETH price is ${market_data["eth_price"]:.2f} with volume of {market_data.get("volume_eth_10m", 0.0):.2f} and volatility of {volatility:.4f}, 
recent price change of {price_change_ticks:.4f} ticks, and I currently hold {self.portfolio["ETH"]:.1f} ETH and {self.portfolio["USDC"]:.0f} USDC.

TRADING HISTORY & PERFORMANCE:
{historical_context}

PERFORMANCE INSIGHTS:
{performance_insights}

MARKET PATTERN ANALYSIS:
{market_insights}

Based on this historical data and performance analysis, what trading action should I take on Uniswap?"""

        
        # Estimate token count for the prompt
        prompt_token_count = self._estimate_token_count(user_query)
        
        # Estimate token count for all accumulated decisions
        context_token_count = 0
        for decision in self.context.trading_decisions:
            if decision.reasoning:
                context_token_count += self._estimate_token_count(decision.reasoning)
        
        # Add some tokens for each market state
        market_state_tokens = len(self.context.market_states) * 20  # Approximate token count per market state
        
        # Total estimated tokens
        total_estimated_tokens = prompt_token_count + context_token_count + market_state_tokens
        context_usage_ratio = total_estimated_tokens / self.context_capacity
        
        # Store context usage info for later display
        self.last_context_usage = {
            'tokens': total_estimated_tokens,
            'ratio': context_usage_ratio
        }
        
        logger.info(f"Observation context usage: {total_estimated_tokens} tokens ({context_usage_ratio:.1%})")
        
        try:
            # Get response from either MLX model or Ollama
            if USE_MLX_MODEL:
                # Use the custom-trained MLX model
                system_prompt = "You are a trading assistant that helps users make trading decisions based on market data. You first use chain-of-draft reasoning with short steps, then provide a clear trading action."
                full_prompt = f"{system_prompt}\n\n{user_query}"
                
                # Load the model if not already loaded
                if not hasattr(self, 'mlx_model'):
                    logger.info(f"Loading MLX model {MLX_BASE_MODEL} with adapter {MLX_ADAPTER_PATH}")
                    try:
                        # Check if adapter files exist
                        adapter_path = MLX_ADAPTER_PATH
                        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
                        adapters_path = os.path.join(adapter_path, "adapters.safetensors")
                        
                        logger.info(f"Checking adapter directory at: {adapter_path}")
                        
                        if not os.path.exists(adapter_path):
                            logger.error(f"Adapter directory does not exist: {adapter_path}")
                            raise FileNotFoundError(f"Adapter directory does not exist: {adapter_path}")
                        
                        if not os.path.exists(adapter_config_path):
                            logger.error(f"adapter_config.json not found at: {adapter_config_path}")
                            raise FileNotFoundError(f"adapter_config.json not found at: {adapter_config_path}")
                            
                        if not os.path.exists(adapters_path):
                            logger.error(f"adapters.safetensors not found at: {adapters_path}")
                            raise FileNotFoundError(f"adapters.safetensors not found at: {adapters_path}")
                            
                        logger.info(f"Adapter files found, loading model with adapter path: {adapter_path}")
                        
                        self.mlx_model, self.mlx_tokenizer = load(
                            MLX_BASE_MODEL, 
                            adapter_path=adapter_path
                        )
                    except Exception as e:
                        logger.error(f"Error loading MLX model with adapter: {str(e)}")
                        # Fall back to base model without adapter
                        logger.info(f"Falling back to base model without adapter")
                        self.mlx_model, self.mlx_tokenizer = load(MLX_BASE_MODEL)
                
                # Generate response
                tokens = self.mlx_tokenizer.encode(full_prompt)
                results = generate(
                    self.mlx_model,
                    self.mlx_tokenizer,
                    prompt=full_prompt,
                    max_tokens=500,
                    sampler=make_sampler(temp=0.6)
                )
                # Handle different return types from generate function
                if isinstance(results, list) and isinstance(results[0], dict):
                    response_text = results[0]["generation"]
                elif isinstance(results, str):
                    response_text = results
                else:
                    response_text = str(results)
                logger.info(f"MLX Model Response (Observation): {response_text}")
            elif self.model_provider == 'openrouter':
                # Call OpenRouter API for observation
                start_time = time.time()
                try:
                    response = self.chat_with_openrouter(
                        messages=[
                            {"role": "system", "content": "You are a trading assistant that helps users make trading decisions based on market data. You first use chain-of-draft reasoning with short steps, then provide a clear trading action."},
                            {"role": "user", "content": user_query}
                        ],
                        timeout=90,  # 90 second timeout for trading decisions
                        max_tokens=2000, # 2000 for OpenRouter Grok-4; 500 is suggested for fine-tuned custom chain-of-draft models
                        temperature=0.6
                    )
                    
                    response_text = response.get("message", {}).get("content", "")
                    
                    # Debug: Log full response length and first part
                    logger.info(f"Full OpenRouter response length: {len(response_text)} characters")
                    if len(response_text) > 100:
                        logger.info(f"Response preview: {response_text[:200]}...")
                    else:
                        logger.info(f"Full response: {response_text}")
                    
                    # Note: Don't truncate OpenRouter responses - they don't use RECOMMENDATION format
                    logger.info(f"OpenRouter observation response received in {time.time() - start_time:.2f} seconds")
                except Exception as e:
                    logger.error(f"Error getting OpenRouter observation response: {e}")
                    response_text = "ERROR: Model response timed out"
            else:
                # Call Ollama with timeout protection for observation mode
                start_time = time.time()
                try:
                    # Use our safe chat method with timeout - NO SYSTEM PROMPT for trader-qwen
                    response = self.chat_with_timeout(
                        model=OLLAMA_MODEL,
                        messages=[
                            {
                                "role": "user",
                                "content": user_query
                            }
                        ],
                        timeout=90  # 90 second timeout for trading decisions
                    )
                    
                    response_text = response.get("message", {}).get("content", "")
                    
                    # Truncate response after RECOMMENDATION to avoid garbage content (Ollama models only)
                    recommendation_match = re.search(r"RECOMMENDATION:.*$", response_text, re.MULTILINE)
                    if recommendation_match:
                        # Keep only up to the end of the recommendation line
                        end_pos = recommendation_match.end()
                        response_text = response_text[:end_pos]
                        
                    logger.info(f"Observation response received in {time.time() - start_time:.2f} seconds")
                except Exception as e:
                    logger.error(f"Error getting observation response: {e}")
                    response_text = "ERROR: Model response timed out"
            
            logger.info(f"\n--- OBSERVATION #{len(self.context.trading_decisions)+1} ---\n{response_text}\n----------------------------")
            
            # Print response to console so user can see it
            print(f"\n--- OBSERVATION #{len(self.context.trading_decisions)+1} ---")
            print(response_text)
            print("----------------------------")
            
            # Extract trading decision - different parsing for different model types (observation mode)
            if USE_MLX_MODEL or self.model_provider != 'openrouter':
                # Look for RECOMMENDATION pattern (MLX/Ollama format)
                canary_match = re.search(r"RECOMMENDATION:\s+APE\s+(IN|OUT|NEUTRAL)(?:\s+([\d\.]+|part of|some)\s*(?:ETH)?)?", response_text)
            else:
                # Parse natural language responses from OpenRouter models (e.g., Grok-4)
                parsed_decision = self._parse_natural_language_trading_decision(response_text)
                canary_match = parsed_decision
            
            if canary_match:
                action = canary_match.group(1).upper()
                
                # Get amount string if available
                amount_str = canary_match.group(2) if len(canary_match.groups()) >= 2 else None
                current_time = int(time.time())
                
                # Convert action to trade parameters
                if action == "OUT":
                    trade_type = "ETH_TO_USDC"
                    
                    # Determine amount based on the amount string
                    if amount_str and amount_str.replace('.', '', 1).isdigit():
                        # If it's a numeric value like "0.2"
                        amount = float(amount_str)
                    elif amount_str and "part of" in amount_str:
                        # If it's "part of ETH"
                        amount = 0.2 * self.portfolio["ETH"]  # Use 20% as default for "part of"
                    else:
                        # Default amount if not specified
                        amount = 0.2 * self.portfolio["ETH"]
                    
                elif action == "IN":
                    trade_type = "USDC_TO_ETH"
                    
                    # Determine amount based on the amount string
                    market_data = self.get_market_data()
                    if amount_str and amount_str.replace('.', '', 1).isdigit():
                        # If it's a numeric value like "0.2"
                        usdc_amount = float(amount_str) * market_data["eth_price"]
                        amount = usdc_amount / market_data["eth_price"]
                    elif amount_str and "some" in amount_str:
                        # If it's "some ETH"
                        usdc_amount = 0.2 * self.portfolio["USDC"]
                        amount = usdc_amount / market_data["eth_price"]
                    else:
                        # Default amount if not specified
                        usdc_amount = 0.2 * self.portfolio["USDC"]
                        amount = usdc_amount / market_data["eth_price"]
                    
                else:  # NEUTRAL
                    # For NEUTRAL, don't trade
                    decision = TradingDecision(
                        timestamp=current_time,
                        decision_type="LLM_DECISION",
                        trade_type=None,
                        amount=0.0,
                        reasoning=response_text,
                        market_state=self.context.market_states[-1],
                        portfolio_before=self.portfolio.copy()
                    )
                    self.context.add_trading_decision(decision)
                    return
                
                # Add minimum trade size check
                if trade_type == "ETH_TO_USDC" and amount < 0.00001:
                    logger.warning(f"ETH amount {amount:.6f} too small for trade, minimum is 0.00001 ETH")
                    
                    # Record as observation without trade
                    decision = TradingDecision(
                        timestamp=int(time.time()),
                        decision_type="OBSERVATION",
                        trade_type=None,
                        amount=0.0,
                        reasoning=response_text,
                        market_state=self.context.market_states[-1],
                        portfolio_before=self.portfolio.copy()
                    )
                    self.context.add_trading_decision(decision)
                    return
                    
                elif trade_type == "USDC_TO_ETH" and amount * market_data["eth_price"] < 0.01:
                    logger.warning(f"USDC amount {amount * market_data['eth_price']:.6f} too small for trade, minimum is 0.01 USDC")
                    
                    # Record as observation without trade
                    decision = TradingDecision(
                        timestamp=int(time.time()),
                        decision_type="OBSERVATION",
                        trade_type=None,
                        amount=0.0,
                        reasoning=response_text,
                        market_state=self.context.market_states[-1],
                        portfolio_before=self.portfolio.copy()
                    )
                    self.context.add_trading_decision(decision)
                    return
                
                # Create trading decision
                decision = TradingDecision(
                    timestamp=current_time,
                    decision_type="LLM_DECISION",
                    trade_type=trade_type,
                    amount=amount,
                    reasoning=response_text,
                    market_state=self.context.market_states[-1],
                    portfolio_before=self.portfolio.copy()
                )
                
                # Execute trade
                trade_result = self.execute_trade(
                    trade_type=trade_type,
                    amount=amount,
                    reason=response_text
                )
                
                if trade_result:
                    # Update decision with results
                    decision.portfolio_after = self.portfolio.copy()
                    decision.success = trade_result["status"] == "success"
                    decision.profit_loss = self.calculate_decision_performance(decision)
                    
                    # Add to context
                    self.context.add_trading_decision(decision)
                    return
            else:
                # No recommendation found, record as observation only
                decision = TradingDecision(
                    timestamp=int(time.time()),
                    decision_type="OBSERVATION",
                    trade_type=None,
                    amount=0.0,
                    reasoning=response_text,
                    market_state=self.context.market_states[-1],
                    portfolio_before=self.portfolio.copy()
                )
                self.context.add_trading_decision(decision)
            
        except Exception as e:
            logger.error(f"Error in observation analysis: {e}")
            traceback.print_exc()
    
    def _generate_initial_strategy(self):
        """Generate an initial trading strategy based on accumulated observations"""
        if not hasattr(self, 'ollama_client') and not hasattr(self, 'openrouter_client') and not USE_MLX_MODEL:
            return
        
        # Get market trends
        market_trends = self.context.get_market_trend(minutes=60)
        
        # Get all accumulated observations for comprehensive analysis
        all_observations = list(self.context.trading_decisions)
        observation_count = len(all_observations)
        
        # Get current market data for formatted query
        market_data = self.get_market_data()
        
        # Calculate ETH volatility
        volatility = market_trends["volatility"] * 100
        
        # Calculate recent price change in ticks (scaled to match training data format)
        price_change_ticks = market_data.get("price_change_pct_10m", 0.0) * 10
        
        # Prepare user query in the format of the trading data
        user_query = f"""Given ETH price is ${market_data["eth_price"]:.2f} with volume of {market_data.get("volume_eth_10m", 0.0):.2f} and volatility of {volatility:.4f}, 
        recent price change of {price_change_ticks:.4f} ticks, and I currently hold {self.portfolio["ETH"]:.1f} ETH and {self.portfolio["USDC"]:.0f} USDC, 
        what trading action should I take on Uniswap?"""

        
        logger.info(f"Generating initial strategy based on {observation_count} observations")
        
        try:
            # Use LLM to generate trading strategy
            if USE_MLX_MODEL:
                # ... existing MLX model code ...
                pass
            elif self.model_provider == 'openrouter':
                # Call OpenRouter API for strategy generation
                start_time = time.time()
                try:
                    response = self.chat_with_openrouter(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a trading assistant that helps users make trading decisions based on market data. You first use chain-of-draft reasoning with short steps, then provide a clear trading action.",
                            },
                            {
                                "role": "user",
                                "content": user_query
                            }
                        ],
                        timeout=60,  # 60 second timeout for strategy generation
                        max_tokens=2000,  # 2000 for OpenRouter Grok-4; 500 is suggested for fine-tuned custom chain-of-draft models
                        temperature=0.6
                    )
                    
                    strategy_text = response.get("message", {}).get("content", "")
                    
                    # Note: Don't truncate OpenRouter responses - they don't use RECOMMENDATION format
                    logger.info(f"OpenRouter strategy generation completed in {time.time() - start_time:.2f} seconds")
                except Exception as e:
                    logger.error(f"Error generating strategy via OpenRouter: {e}")
                    return
            else:
                # Call Ollama with timeout protection
                start_time = time.time()
                try:
                    # Use our safe chat method with timeout - NO SYSTEM PROMPT for trader-qwen
                    response = self.chat_with_timeout(
                        model=OLLAMA_MODEL, 
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a trading assistant that helps users make trading decisions based on market data. You first use chain-of-draft reasoning with short steps, then provide a clear trading action.",
                            },
                            {
                                "role": "user",
                                "content": user_query
                            }
                        ],
                        timeout=60  # 60 second timeout for strategy generation
                    )
                    
                    strategy_text = response.get("message", {}).get("content", "")
                    
                    # Truncate response after RECOMMENDATION to avoid garbage content (Ollama models only)
                    recommendation_match = re.search(r"RECOMMENDATION:.*$", strategy_text, re.MULTILINE)
                    if recommendation_match:
                        # Keep only up to the end of the recommendation line
                        end_pos = recommendation_match.end()
                        strategy_text = strategy_text[:end_pos]
                        
                    logger.info(f"Strategy generation completed in {time.time() - start_time:.2f} seconds")
                except Exception as e:
                    logger.error(f"Error generating strategy: {e}")
                    return
            
            logger.info(f"Initial strategy generated: {strategy_text}")
            
            # Print strategy to console so user can see it
            print(f"\n--- INITIAL STRATEGY ---")
            print(strategy_text)
            print("------------------------")
            
            # Extract strategy based on Canary words - only for MLX and Ollama models
            if USE_MLX_MODEL or self.model_provider != 'openrouter':
                # Look for APE pattern (MLX/Ollama format)
                canary_match = re.search(r"####\s+APE\s+(IN|OUT|NEUTRAL)", strategy_text)
            else:
                # OpenRouter models don't use the APE format
                canary_match = None
                
            if canary_match:
                action = canary_match.group(1)
                strategy_summary = f"Initial strategy: APE {action}"
                
                # Set the strategy in the context
                self.context.current_strategy = strategy_summary
                self.context.strategy_duration = self.min_strategy_duration
                self.context.strategy_start_time = int(time.time())
                
                # Log the strategy
                logger.info(f"Initialized strategy: {strategy_summary}")
                
                # Create a trading decision for the initial strategy
                decision = TradingDecision(
                    timestamp=int(time.time()),
                    decision_type="STRATEGY",
                    trade_type=None,
                    amount=0.0,
                    reasoning=strategy_text,
                    market_state=self.context.market_states[-1],
                    portfolio_before=self.portfolio.copy()
                )
                self.context.add_trading_decision(decision)
            else:
                logger.warning("No clear Canary word found in initial strategy")
                self.context.current_strategy = "Default strategy after observation"
                self.context.strategy_duration = self.min_strategy_duration
                self.context.strategy_start_time = int(time.time())
            
        except Exception as e:
            logger.error(f"Error generating initial strategy: {e}")
            traceback.print_exc()
            self.context.current_strategy = "Default strategy after observation"
            self.context.strategy_duration = self.min_strategy_duration
            self.context.strategy_start_time = int(time.time())

if __name__ == "__main__":
    import asyncio
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Uniswap v4 Stateful Trading Agent')
    parser.add_argument('--target-allocation', type=float, default=DEFAULT_ETH_ALLOCATION,
                      help='Target ETH allocation (0-1, default: 0.5)')
    parser.add_argument('--iterations', type=int, default=None,
                      help='Number of trading iterations (default: run indefinitely)')
    parser.add_argument('--test-mode', action='store_true',
                      help='Enable test mode with reduced context capacity to test summarization')
    parser.add_argument('--observe', type=int, default=None,
                      help='Run in observation mode for specified context window size in tokens before trading (gathers market data without executing trades)')
    parser.add_argument('--observe-time', type=int, default=None,
                      help='Run in observation mode for specified minutes before trading (gathers market data without executing trades)')
    parser.add_argument('--observe-cycles', type=int, default=None,
                      help='Run in observation mode for specified number of observation cycles before trading (gathers market data without executing trades)')
    args = parser.parse_args()
    
    async def main():
        # Initialize trading agent with command line arguments
        agent = UniswapV4StatefulTradingAgent(
            target_eth_allocation=args.target_allocation,
            test_mode=args.test_mode
        )
        
        # Run the trading loop with observation mode if specified
        if args.observe or args.observe_time or args.observe_cycles:
            logger.info(f"Starting observation mode before trading")
            if args.observe:
                logger.info(f"Will observe until context reaches {args.observe} tokens")
            elif args.observe_time:
                logger.info(f"Will observe for {args.observe_time} minutes")
            elif args.observe_cycles:
                logger.info(f"Will observe for {args.observe_cycles} cycles")
            
            agent.run_observation_mode(
                target_tokens=args.observe,
                target_minutes=args.observe_time,
                target_cycles=args.observe_cycles
            )
        
        # Run the regular trading loop
        agent.run(iterations=args.iterations)
    
    # Run the async main function
    asyncio.run(main()) 