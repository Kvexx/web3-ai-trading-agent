"""
Uniswap V4 Trading Agent
------------------------
Trading agent that uses UniswapV4Swapper for executing trades
using LLM-based trading decisions
"""

import os
import sys
import time
import json
import logging
import argparse
import re
import asyncio
from typing import Dict, Tuple, List, Any, Optional
from web3 import Web3
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
from decimal import Decimal
import ollama
from datetime import datetime
from rich import box
import concurrent.futures

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

# Import the Uniswap V4 Swapper
from uniswap_v4_swapper import UniswapV4Swapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

console = Console()

# Import configuration
from config import (
    BASE_RPC_URL,
    UNISWAP_V4_POOL_MANAGER,
    UNISWAP_V4_POSITION_MANAGER,
    UNISWAP_V4_UNIVERSAL_ROUTER,
    UNISWAP_V4_STATE_VIEW,
    BASE_USDC_ADDRESS,
    ETH_ADDRESS,
    WETH_ADDRESS,
    LOCAL_RPC_URL,
    USE_FORK,
    OLLAMA_MODEL,
    OLLAMA_URL,
    PRIVATE_KEY,
    TRADE_INTERVAL,
    REBALANCE_THRESHOLD,
    DEFAULT_ETH_ALLOCATION,
    UNISWAP_V4_ETH_USDC_POOL,
    USE_MLX_MODEL,
    MLX_BASE_MODEL,
    MLX_ADAPTER_PATH,
    MODEL_KEY,
    AVAILABLE_MODELS,
    CONTEXT_WARNING_THRESHOLD,
    get_context_capacity,
)

# Import MLX modules if we're using the custom model
if USE_MLX_MODEL:
    try:
        import mlx.core as mx
        from mlx_lm.utils import load_model, generate_step
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
        
        # Use the absolute path for logging instead of reporting the relative one
        adapter_path = os.path.abspath(MLX_ADAPTER_PATH) if isinstance(MLX_ADAPTER_PATH, str) else MLX_ADAPTER_PATH
        logger.info(f"MLX modules loaded, will use custom-trained model from {adapter_path}")
    except ImportError as e:
        logger.error(f"Failed to import MLX modules: {str(e)}")
        logger.error("Make sure MLX is installed and in your Python path.")
        logger.error("Falling back to Ollama model.")
        USE_MLX_MODEL = False

class UniswapV4TradingAgent:
    """
    Trading agent that uses the UniswapV4Swapper to execute trades,
    making decisions using a local LLM via Ollama
    """
    
    def __init__(self, private_key=None, target_eth_allocation=DEFAULT_ETH_ALLOCATION):
        """
        Initialize the trading agent
        
        Args:
            private_key: Ethereum private key for signing transactions
            target_eth_allocation: Target ETH allocation (0-1)
        """
        # Initialize web3 connection
        try:
            # Choose RPC URL based on USE_FORK setting
            rpc_url = LOCAL_RPC_URL if USE_FORK else BASE_RPC_URL
            environment = "Foundry Anvil fork" if USE_FORK else "BASE mainnet"
            
            # Use HTTP Provider with a sensible timeout
            self.w3 = Web3(Web3.HTTPProvider(
                rpc_url,
                request_kwargs={'timeout': 30.0}  # 30 second timeout for RPC requests
            ))
            
            # Check connection
            if not self.w3.is_connected():
                logger.error(f"Could not connect to {environment} at {rpc_url}")
                raise ConnectionError(f"Could not connect to {environment} at {rpc_url}")
                
            logger.info(f"Connected to {environment}: {rpc_url}")
            logger.info(f"Current block number: {self.w3.eth.block_number}")
        except Exception as e:
            logger.error(f"Error initializing Web3: {e}")
            raise
        
        # Use the first test account from Foundry Anvil if no private key provided
        if not private_key:
            self.private_key = PRIVATE_KEY
            self.account = self.w3.eth.account.from_key(self.private_key)
            self.address = self.account.address
        else:
            self.private_key = private_key
            self.account = self.w3.eth.account.from_key(self.private_key)
            self.address = self.account.address
            
        logger.info(f"Using account: {self.address}")
        
        # Configuration for the swapper
        self.swapper_config = {
            'BASE_USDC_ADDRESS': BASE_USDC_ADDRESS,
            'ETH_ADDRESS': ETH_ADDRESS,
            'WETH_ADDRESS': WETH_ADDRESS,
            'UNISWAP_V4_UNIVERSAL_ROUTER': UNISWAP_V4_UNIVERSAL_ROUTER,
            'UNISWAP_V4_POSITION_MANAGER': UNISWAP_V4_POSITION_MANAGER,
            'UNISWAP_V4_STATE_VIEW': UNISWAP_V4_STATE_VIEW,
            'UNISWAP_V4_POOL_MANAGER': UNISWAP_V4_POOL_MANAGER,
        }
        
        # Initialize the UniswapV4Swapper
        self.swapper = UniswapV4Swapper(self.w3, self.private_key, self.swapper_config)
        
        trading_mode = "Paper Trading (Foundry Fork)" if USE_FORK else "⚠️ LIVE TRADING (Real BASE Mainnet)"
        logger.info(f"Trading mode: {trading_mode}")
        
        # Initialize portfolio balances
        self.portfolio = self.get_actual_balances()
        
        # Set target ETH allocation (as decimal: 0.5 = 50%)
        self.target_eth_allocation = target_eth_allocation
        
        # Keep track of trades
        self.trade_history = []
        
        # For tracking market data
        self.price_history = []
        
        # Last rebalance time
        self.last_rebalance_time = 0
        
        # Small trade size
        self.small_eth_trade_size = 0.01  # 0.01 ETH per trade
        
        # Log model information
        logger.info(f"Using model: {OLLAMA_MODEL if not USE_MLX_MODEL else MLX_BASE_MODEL} (via key: {MODEL_KEY})")

        # Initialize LLM
        self._initialize_llm()
        
        # Parameters for rebalancing
        self.rebalance_interval = 60 * 60  # 1 hour minimum between rebalances
        self.rebalance_threshold = REBALANCE_THRESHOLD
    
    def _initialize_llm(self):
        """Unified method to initialize either MLX or Ollama LLM"""
        if USE_MLX_MODEL:
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
                    self.mlx_model, self.mlx_tokenizer = load(MLX_BASE_MODEL)
                    logger.info("MLX base model loaded successfully")
                    return True
                except Exception as e:
                    logger.error(f"Error loading MLX base model: {str(e)}")
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
                    _ = self.ollama_client.chat(model=OLLAMA_MODEL, messages=[
                        {"role": "system", "content": "You are a trading assistant."},
                        {"role": "user", "content": "Hello"}
                    ])
                    logger.info(f"Model preloaded in {time.time() - preload_start:.2f} seconds")
                    logger.info(f"Ollama initialization completed in {time.time() - test_start:.2f} seconds. Ready for trading.")
                    return True
                except Exception as e:
                    logger.error(f"Error testing Ollama connection: {e}")
                    return False
            except Exception as e:
                logger.error(f"Error initializing Ollama client: {e}")
                logger.info("Will initialize on first request")
                return False
    
    def get_actual_balances(self):
        """Get actual token balances from the swapper"""
        try:
            balances = self.swapper.get_balances()
            logger.info(f"Retrieved on-chain balances: {balances['ETH']:.6f} ETH, {balances['USDC']:.2f} USDC")
            return balances
        except Exception as e:
            logger.error(f"Error fetching on-chain balances: {e}. Using default balances.")
            # Default values if there's an error
            return {"ETH": 0.5, "USDC": 0.0}
    
    def get_market_data(self):
        """Fetch current market data for ETH-USDC pair"""
        try:
            # Get current ETH price
            eth_price = self.swapper.get_eth_price()
            
            # Fetch recent swap events to calculate volume
            volume_data = self.fetch_recent_swap_events(minutes=10)
            
            # Create market data
            market_data = {
                "eth_price": eth_price,  # Will raise TypeError if None
                "price_change_pct_10m": volume_data.get("price_change_pct", 0.0),
                "volume_eth_10m": volume_data.get("volume_eth", 0.0),
                "volume_usdc_10m": volume_data.get("volume_usdc", 0.0),
                "swap_count_10m": volume_data.get("swap_count", 0),
                "timestamp": int(time.time())
            }
                
            return market_data
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            raise  # Re-raise the exception instead of returning fallback data
    
    def fetch_recent_swap_events(self, minutes=10):
        """
        Fetch recent swap events from the pool to calculate volume and price changes
        
        Args:
            minutes: Number of minutes to look back for events
            
        Returns:
            dict: Volume and price data
        """
        try:
            # Start overall timing
            overall_start_time = time.time()
            
            # Use a cache for recent swap events - extend cache time to 30 seconds
            if hasattr(self, '_swap_events_cache') and self._swap_events_cache.get('expiry', 0) > time.time():
                logger.debug("Using cached swap events data")
                return self._swap_events_cache['data']
                
            # Calculate block range (approximately 2 seconds per block)
            current_block = self.w3.eth.block_number
            # Use 150 blocks for faster processing
            start_block = max(0, current_block - 150)
            end_block = current_block  # End block is the current block
            
            # Use a small block range for faster processing
            block_chunk_size = 150
            
            # Get the pool ID from our swapper
            pool_id = self.swapper.pool_id
            
            # ABI for the Swap event
            swap_event_abi = {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "internalType": "bytes32", "name": "id", "type": "bytes32"},
                    {"indexed": True, "internalType": "address", "name": "sender", "type": "address"},
                    {"indexed": False, "internalType": "int128", "name": "amount0", "type": "int128"},
                    {"indexed": False, "internalType": "int128", "name": "amount1", "type": "int128"},
                    {"indexed": False, "internalType": "uint160", "name": "sqrtPriceX96", "type": "uint160"},
                    {"indexed": False, "internalType": "uint128", "name": "liquidity", "type": "uint128"},
                    {"indexed": False, "internalType": "int24", "name": "tick", "type": "int24"},
                    {"indexed": False, "internalType": "uint24", "name": "fee", "type": "uint24"}
                ],
                "name": "Swap",
                "type": "event"
            }
            
            # Create a contract instance for the pool manager
            pool_manager = self.w3.eth.contract(
                address=self.w3.to_checksum_address(UNISWAP_V4_POOL_MANAGER),
                abi=[swap_event_abi]
            )
            
            # Get the Swap event signature
            swap_event_signature = self.w3.keccak(text="Swap(bytes32,address,int128,int128,uint160,uint128,int24,uint24)").hex()
            
            fetch_start_time = time.time()
            logger.info(f"Fetching swap events from blocks {start_block} to {end_block}")
            
            # Calculate chunks for parallel fetching
            chunks = []
            current_start = start_block
            while current_start < end_block:
                current_end = min(current_start + block_chunk_size - 1, end_block)
                # Only add chunk if it spans more than one block
                if current_end > current_start:
                    chunks.append((current_start, current_end))
                current_start = current_end + 1
            
            # If no chunks were created (shouldn't happen with above logic), create one chunk
            if not chunks:
                chunks = [(start_block, end_block)]
            
            # Define a function to fetch a single chunk of logs
            def fetch_chunk(chunk_range):
                chunk_start, chunk_end = chunk_range
                chunk_fetch_start = time.time()
                try:
                    logs = self.w3.eth.get_logs({
                        'address': self.w3.to_checksum_address(UNISWAP_V4_POOL_MANAGER),
                        'fromBlock': chunk_start,
                        'toBlock': chunk_end,
                        'topics': [
                            swap_event_signature,
                            "0x" + pool_id[2:].ljust(64, '0')  # First indexed param is poolId, need to pad to 32 bytes
                        ]
                    })
                    chunk_duration = time.time() - chunk_fetch_start
                    logger.info(f"Fetched chunk {chunk_start}-{chunk_end}: {len(logs)} events in {chunk_duration:.2f}s")
                    return logs
                except Exception as e:
                    logger.warning(f"Error fetching chunk {chunk_start}-{chunk_end}: {e}")
                    return []  # Return empty list on error
            
            # Use thread pool to fetch chunks in parallel with more workers
            all_logs = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(chunks), 8)) as executor:
                # Submit all chunks to the executor
                future_to_chunk = {executor.submit(fetch_chunk, chunk): chunk for chunk in chunks}
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    try:
                        logs = future.result()
                        all_logs.extend(logs)
                    except Exception as e:
                        logger.error(f"Exception processing chunk {chunk}: {e}")
            
            fetch_duration = time.time() - fetch_start_time
            logger.info(f"Fetched all chunks in {fetch_duration:.2f}s")
            
            # Start timing for log processing
            process_start_time = time.time()
            
            # Parse the logs in parallel
            def process_log(log):
                try:
                    return pool_manager.events.Swap().process_log(log)
                except Exception as e:
                    logger.error(f"Error processing swap log: {e}")
                    return None
            
            # Process logs in parallel
            swaps = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                future_to_log = {executor.submit(process_log, log): log for log in all_logs}
                for future in concurrent.futures.as_completed(future_to_log):
                    try:
                        swap = future.result()
                        if swap:
                            swaps.append(swap)
                    except Exception as e:
                        logger.error(f"Error processing swap log: {e}")
            
            process_duration = time.time() - process_start_time
            logger.info(f"Processed {len(swaps)} swap events in {process_duration:.2f}s")
            
            # Calculate volumes and price changes
            volume_eth = 0.0
            volume_usdc = 0.0
            price_start = None
            price_end = None
            
            for i, swap in enumerate(swaps):
                try:
                    # Get amounts from the swap event (absolute values)
                    amount0 = abs(int(swap['args']['amount0'])) / 1e18  # ETH has 18 decimals
                    amount1 = abs(int(swap['args']['amount1'])) / 1e6   # USDC has 6 decimals
                    
                    # Add to volume
                    volume_eth += amount0
                    volume_usdc += amount1
                    
                    # Get price data
                    sqrt_price_x96 = int(swap['args']['sqrtPriceX96'])
                    price = self.calculate_price_from_sqrt_price_x96(sqrt_price_x96)
                    
                    # Get first and last prices for calculating price change
                    if i == 0:
                        price_start = price
                    if i == len(swaps) - 1:
                        price_end = price
                except Exception as e:
                    logger.error(f"Error calculating volumes from swap: {e}")
                    continue
            
            # Calculate price change percentage
            price_change_pct = 0.0
            if price_start and price_end and price_start > 0:
                price_change_pct = ((price_end - price_start) / price_start) * 100
            
            # Create result data
            result_data = {
                "volume_eth": volume_eth,
                "volume_usdc": volume_usdc,
                "swap_count": len(swaps),
                "price_change_pct": price_change_pct
            }
            
            # Cache the results for 60 seconds
            self._swap_events_cache = {
                'data': result_data,
                'expiry': time.time() + 60  # Cache for 60 seconds
            }
            
            # Log overall timing
            overall_duration = time.time() - overall_start_time
            logger.info(f"Total swap events fetch and process completed in {overall_duration:.2f}s")
            
            return result_data
            
        except Exception as e:
            logger.error(f"Error fetching swap events: {e}")
            # Return empty data rather than failing
            return {"volume_eth": 0.0, "volume_usdc": 0.0, "swap_count": 0, "price_change_pct": 0.0}
    
    def calculate_price_from_sqrt_price_x96(self, sqrt_price_x96):
        """
        Calculate price from sqrtPriceX96 value using Uniswap V4's formula
        Price = (sqrtPriceX96 / 2^96) ^ 2
        
        Returns:
            float: ETH price in USDC
        """
        try:
            # Convert to Decimal for precise calculation
            sqrt_price = Decimal(sqrt_price_x96)
            two_96 = Decimal(2) ** Decimal(96)
            
            # Calculate sqrtPrice / 2^96
            sqrt_price_adjusted = sqrt_price / two_96
            
            # Square it to get the price
            price = sqrt_price_adjusted * sqrt_price_adjusted
            
            # ETH (18 decimals) to USDC (6 decimals) = multiply by 10^12
            price = price * Decimal(10 ** 12)
            
            return float(price)
        except Exception as e:
            logger.error(f"Error calculating price: {e}")
            return None
    
    def calculate_portfolio_value(self, eth_price):
        """Calculate total portfolio value in USD"""
        eth_value = self.portfolio["ETH"] * eth_price
        usdc_value = self.portfolio["USDC"]
        return eth_value + usdc_value
    
    def get_eth_allocation(self):
        """Get current ETH allocation as percentage of portfolio"""
        market_data = self.get_market_data()
        eth_price = market_data["eth_price"]
        
        total_value = self.calculate_portfolio_value(eth_price)
        if total_value == 0:
            return 0
        
        eth_value = self.portfolio["ETH"] * eth_price
        return (eth_value / total_value) * 100 if total_value > 0 else 0
    
    def should_rebalance(self):
        """
        Determine if portfolio needs rebalancing based on:
        1. Current allocation vs target
        2. Time since last rebalance
        """
        # Get current market data
        market_data = self.get_market_data()
        eth_price = market_data["eth_price"]
        
        # Calculate current ETH allocation
        total_value = self.calculate_portfolio_value(eth_price)
        eth_value = self.portfolio["ETH"] * eth_price
        current_eth_allocation = eth_value / total_value if total_value > 0 else 0
        
        # Calculate the deviation from target
        allocation_deviation = abs(current_eth_allocation - self.target_eth_allocation)
        
        # Check if enough time has passed since last rebalance
        current_time = int(time.time())
        time_since_last_rebalance = current_time - self.last_rebalance_time
        
        logger.debug(f"Current ETH allocation: {current_eth_allocation:.2%}")
        logger.debug(f"Target ETH allocation: {self.target_eth_allocation:.2%}")
        logger.debug(f"Allocation deviation: {allocation_deviation:.2%}")
        logger.debug(f"Time since last rebalance: {time_since_last_rebalance}s")
        
        # Determine if rebalancing is needed
        if allocation_deviation >= self.rebalance_threshold and time_since_last_rebalance >= self.rebalance_interval:
            # Determine trade direction (ETH to USDC or USDC to ETH)
            if current_eth_allocation > self.target_eth_allocation:
                direction = "ETH_TO_USDC"
                
                # Calculate trade amount
                # Use conservative rebalancing
                # We'll use at most 2% of the ETH balance per trade
                amount = min(0.02, allocation_deviation) * self.portfolio["ETH"]
                # For safety, cap at our smallest trade size
                amount = min(amount, 0.05)  # Cap at 0.05 ETH max
                # Make sure we have a minimum trade amount
                amount = max(amount, self.small_eth_trade_size)
            else:
                direction = "USDC_TO_ETH"
                # Calculate how much USDC to convert to ETH
                eth_deficit = (self.target_eth_allocation - current_eth_allocation) * total_value
                
                # More conservative rebalancing
                # We'll use at most 2% of USDC balance per trade, converted to ETH
                max_affordable = self.portfolio["USDC"] * 0.02 / eth_price
                amount = min(max_affordable, 0.05)  # Cap at 0.05 ETH max
                # Make sure we have a minimum trade amount
                amount = max(amount, self.small_eth_trade_size)
            
            return True, direction, amount
        
        return False, None, 0
    
    def calculate_price_trend(self, market_data):
        """
        Calculate price trend based on recent market data
        Returns:
            float: Price trend indicator (positive = upward, negative = downward)
        """
        # In absence of detailed data, just return 0
        return 0.0
    
    def make_trading_decision(self):
        """
        Determine whether to trade based on:
        1. Portfolio rebalancing needs
        2. LLM-based trading signal
        """
        # First check if we need to rebalance
        should_rebalance, direction, amount = self.should_rebalance()
        
        if should_rebalance:
            logger.info(f"Rebalancing portfolio: ETH allocation ({self.get_eth_allocation():.2%}) {'exceeds' if direction == 'ETH_TO_USDC' else 'below'} target ({self.target_eth_allocation:.2%}) by {abs(self.get_eth_allocation() - self.target_eth_allocation):.2%}")
            
            # Update rebalance time
            self.last_rebalance_time = int(time.time())
            
            # If amount is very small, use minimum trade size
            if amount < self.small_eth_trade_size:
                amount = self.small_eth_trade_size
                
            # Log trade parameters
            logger.info(f"Executing rebalancing trade: {direction}, amount: {amount:.6f} ETH")
            
            # Execute the rebalancing trade
            return self.execute_trade(
                trade_type=direction,
                amount=amount,
                reason=f"Rebalancing: {'Selling ETH' if direction == 'ETH_TO_USDC' else 'Buying ETH'} to reach target allocation of {self.target_eth_allocation:.2%}"
            )
        
        # If no rebalancing needed, use LLM for trading decisions
        market_data = self.get_market_data()
        
        # Get price trend
        price_trend = self.calculate_price_trend(market_data)
        
        # Prepare market and portfolio data for LLM
        portfolio_value = self.calculate_portfolio_value(market_data["eth_price"])
        eth_allocation = self.get_eth_allocation()
        
        # Check if we have enough assets to trade at all
        MIN_ETH_FOR_GAS = 0.0005
        MIN_TRADE_SIZE_ETH = 0.0001
        MIN_TRADE_SIZE_USDC = 0.01
        
        # Balance checks for potential trades
        eth_available_for_trade = max(0, self.portfolio["ETH"] - MIN_ETH_FOR_GAS)
        usdc_available_for_trade = self.portfolio["USDC"]
        
        eth_trade_viable = eth_available_for_trade >= MIN_TRADE_SIZE_ETH
        usdc_trade_viable = usdc_available_for_trade >= MIN_TRADE_SIZE_USDC
        
        # Add constraints to the prompt based on balance analysis
        balance_constraints = (
            f"IMPORTANT CONSTRAINTS:\n"
            f"- ETH Available for trading: {eth_available_for_trade:.6f} ETH (${eth_available_for_trade * market_data['eth_price']:.2f})\n"
            f"- USDC Available for trading: {usdc_available_for_trade:.2f} USDC\n"
        )
        
        if not eth_trade_viable:
            balance_constraints += "- WARNING: ETH balance is too low to execute ETH_TO_USDC trades due to gas requirements\n"
        
        if not usdc_trade_viable:
            balance_constraints += "- WARNING: USDC balance is too low to execute USDC_TO_ETH trades\n"
        
        # Prepare prompt for LLM
        prompt = f"""
You are an advanced trading assistant for a Uniswap v4 ETH/USDC trading bot. Given the current market data and portfolio, determine if a trade should be executed.

Current Market Data:
- ETH Price: ${market_data["eth_price"]:.2f}
- Price Change (10 min): {market_data.get("price_change_pct_10m", 0.0):.2f}%
- Trading Volume (10 min): {market_data.get("volume_eth_10m", 0.0):.4f} ETH / ${market_data.get("volume_usdc_10m", 0.0):.2f} USDC
- Number of Swaps (10 min): {market_data.get("swap_count_10m", 0)}

Portfolio Status:
- ETH Balance: {self.portfolio["ETH"]:.4f} ETH (${self.portfolio["ETH"] * market_data["eth_price"]:.2f})
- USDC Balance: {self.portfolio["USDC"]:.2f} USDC
- Total Value: ${portfolio_value:.2f}
- Current ETH Allocation: {eth_allocation:.2f}%
- Target ETH Allocation: {self.target_eth_allocation * 100:.2f}%

{balance_constraints}

Trade Constraints:
- Small trade size: {MIN_TRADE_SIZE_ETH}-0.05 ETH per trade
- ETH_TO_USDC requires keeping at least {MIN_ETH_FOR_GAS} ETH for gas
- USDC_TO_ETH requires at least {MIN_TRADE_SIZE_USDC} USDC
- Avoid executing more than 1 ETH in a single trade

Based on this information, decide whether to trade, in which direction, and what percentage of the balance to trade.
You MUST consider the balance constraints above and only recommend trades that can be executed with the available balances.

Format your response exactly like this:
TRADE: YES/NO
DIRECTION: ETH_TO_USDC or USDC_TO_ETH
PERCENTAGE: <number between 1-100>

REASONING: <your detailed reasoning for the decision>
"""
        
        try:
            # Get response from either MLX model or Ollama
            if USE_MLX_MODEL:
                # Use the custom-trained MLX model
                system_prompt = "You are an AI trading assistant that makes ETH/USDC trading decisions based on market data."
                full_prompt = f"{system_prompt}\n\n{prompt}"
                
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
            else:
                # Use Ollama as before
                response = self.ollama_client.chat(model=OLLAMA_MODEL, messages=[
                    {
                        "role": "system", 
                        "content": "You are an AI trading assistant that makes ETH/USDC trading decisions based on market data."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ])
                response_text = response["message"]["content"]
                logger.info(f"Ollama Response: {response_text}")
            
            # Parse LLM response
            if "TRADE: YES" in response_text:
                # Extract direction
                if "DIRECTION: ETH_TO_USDC" in response_text:
                    trade_type = "ETH_TO_USDC"
                    # Check if ETH trade is viable
                    if not eth_trade_viable:
                        logger.warning("LLM suggested ETH_TO_USDC but ETH balance is insufficient for trade and gas")
                        return None
                    available_balance = eth_available_for_trade
                elif "DIRECTION: USDC_TO_ETH" in response_text:
                    trade_type = "USDC_TO_ETH"
                    # Check if USDC trade is viable
                    if not usdc_trade_viable:
                        logger.warning("LLM suggested USDC_TO_ETH but USDC balance is insufficient")
                        return None
                    available_balance = self.portfolio["USDC"] / market_data["eth_price"]
                else:
                    logger.warning("LLM did not specify valid trade direction")
                    return None
                
                # Extract percentage
                percentage_match = re.search(r"PERCENTAGE: (\d+)", response_text)
                if percentage_match:
                    percentage = int(percentage_match.group(1)) / 100
                else:
                    logger.warning("LLM did not specify valid percentage")
                    return None
                
                # Extract reasoning
                reasoning_match = re.search(r"REASONING: (.*?)(?=$|TRADE:)", response_text, re.DOTALL)
                reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
                
                # Calculate trade amount based on actual available balance
                # Make sure we're using a reasonable percentage that doesn't exceed trade limits
                amount = percentage * available_balance
                
                # Apply minimum and maximum trade sizes
                if trade_type == "ETH_TO_USDC":
                    # Cap at 0.05 ETH max for regular trades
                    amount = min(amount, 0.05)
                    # Ensure minimum viable trade size
                    if amount < MIN_TRADE_SIZE_ETH:
                        amount = MIN_TRADE_SIZE_ETH if available_balance >= MIN_TRADE_SIZE_ETH else available_balance
                else:  # USDC_TO_ETH
                    # Calculate equivalent ETH value
                    usdc_amount = amount * market_data["eth_price"]
                    # Cap at max USDC amount
                    usdc_amount = min(usdc_amount, self.portfolio["USDC"] * 0.95)
                    # Convert back to ETH equivalent
                    amount = usdc_amount / market_data["eth_price"]
                    # Ensure minimum viable trade size in ETH terms
                    if amount < MIN_TRADE_SIZE_ETH:
                        if available_balance >= MIN_TRADE_SIZE_ETH:
                            amount = MIN_TRADE_SIZE_ETH
                        else:
                            logger.warning(f"Trade amount too small: {amount:.8f} ETH")
                            return None
                
                # Log trade parameters
                eth_amount = amount if trade_type == "ETH_TO_USDC" else amount
                usdc_amount = amount * market_data["eth_price"] if trade_type == "ETH_TO_USDC" else amount * market_data["eth_price"]
                logger.info(f"Executing LLM-guided trade: {trade_type}, amount: {amount:.6f} ETH (${usdc_amount:.2f} USDC), percentage suggested: {percentage:.2%}")
                
                # Execute trade
                return self.execute_trade(
                    trade_type=trade_type,
                    amount=amount,
                    reason=reasoning
                )
            else:
                logger.info("LLM decided not to trade")
                return None
        
        except Exception as e:
            logger.error(f"Error getting LLM decision: {e}")
            return None
    
    def execute_trade(self, trade_type, amount, reason=""):
        """
        Execute a trade using the UniswapV4Swapper
        
        Args:
            trade_type: "ETH_TO_USDC" or "USDC_TO_ETH"
            amount: Amount of ETH to swap (for both directions)
            reason: Reason for trade
        
        Returns:
            dict: Trade details
        """
        try:
            print("-" * 50)
            print(f"EXECUTING TRADE: {trade_type}")
            print(f"Reasoning: {reason}")
            print("-" * 50)
            
            # Get market data for price info
            market_data = self.get_market_data()
            eth_price = market_data["eth_price"]
            
            # Record start balances
            balances_before = self.portfolio.copy()
            eth_balance_before = self.portfolio["ETH"]
            usdc_balance_before = self.portfolio["USDC"]
            
            print(f"Account: {self.address}")
            print(f"ETH balance: {eth_balance_before:.8f} ETH")
            print(f"USDC balance: {usdc_balance_before:.6f} USDC")
            print("-" * 50)
            
            # Check if we have enough ETH for gas before executing the trade
            MIN_ETH_FOR_GAS = 0.0005
            
            if trade_type == "ETH_TO_USDC":
                # For ETH to USDC, check if we have enough ETH left after the trade for gas
                if eth_balance_before - amount < MIN_ETH_FOR_GAS:
                    adjusted_amount = max(0, eth_balance_before - MIN_ETH_FOR_GAS)
                    if adjusted_amount < 0.0001:  # If too small to trade
                        logger.warning(f"Insufficient ETH for trade and gas. Need to keep {MIN_ETH_FOR_GAS} ETH for gas.")
                        print(f"Transaction aborted: Insufficient ETH for trade and gas. Need to keep {MIN_ETH_FOR_GAS} ETH for gas.")
                        return {
                            "trade_type": trade_type,
                            "amount_in": 0,
                            "amount_out": 0,
                            "price": eth_price,
                            "timestamp": int(time.time()),
                            "tx_hash": "0x" + "0" * 64,
                            "status": "aborted",
                            "reason": "Insufficient ETH for trade and gas"
                        }
                    logger.info(f"Adjusting ETH amount from {amount:.8f} to {adjusted_amount:.8f} to reserve gas")
                    amount = adjusted_amount
            
            # Execute the trade via the swapper
            if trade_type == "ETH_TO_USDC":
                # For ETH to USDC, amount is the ETH amount
                print(f"Swapping {amount:.8f} ETH to USDC via Universal Router")
                result = self.swapper.swap_eth_to_usdc(amount)
            else:  # USDC_TO_ETH
                # For USDC to ETH, calculate USDC amount
                # Reduce USDC amount to account for potential approval costs or errors
                usdc_amount = min(amount * eth_price, usdc_balance_before * 0.95)
                if usdc_amount < 0.01:  # Minimum viable trade size
                    logger.warning(f"USDC amount {usdc_amount:.6f} too small for trade")
                    print(f"Transaction aborted: USDC amount {usdc_amount:.6f} too small for trade")
                    return {
                        "trade_type": trade_type,
                        "amount_in": 0,
                        "amount_out": 0,
                        "price": eth_price,
                        "timestamp": int(time.time()),
                        "tx_hash": "0x" + "0" * 64,
                        "status": "aborted",
                        "reason": "USDC amount too small for trade"
                    }
                print(f"Swapping {usdc_amount:.6f} USDC to ETH via Universal Router")
                result = self.swapper.swap_usdc_to_eth(usdc_amount)
            
            if result["success"]:
                # Update portfolio with actual values
                self.portfolio = self.get_actual_balances()
                
                # Calculate the actual amount received
                if trade_type == "ETH_TO_USDC":
                    amount_in = result["eth_spent"]
                    amount_out = result["usdc_received"]
                    print(f"Transaction sent: {result['tx_hash']}")
                    print("Transaction successful!")
                else:  # USDC_TO_ETH
                    amount_in = result["usdc_spent"]
                    # Use pure_eth_received if available (includes gas cost adjustment)
                    amount_out = result.get("eth_received", 0)
                    pure_amount = result.get("pure_eth_received", amount_out)
                    gas_cost = result.get("gas_cost_eth", 0)
                    
                    print(f"Transaction sent: {result['tx_hash']}")
                    print("Transaction successful!")
                    print(f"Gas cost: {gas_cost:.8f} ETH")
                    
                    # If we have the pure amount, show both
                    if "pure_eth_received" in result:
                        print(f"ETH received (before gas): {result['pure_eth_received']:.8f} ETH")
                        print(f"ETH received (after gas): {result['eth_received']:.8f} ETH")
                
                tx_status = "success"
                tx_hash = result["tx_hash"]
            else:
                logger.error(f"Trade failed: {result.get('error', 'Unknown error')}")
                print(f"Transaction failed: {result.get('error', 'Unknown error')}")
                
                # Set failed trade parameters
                tx_status = "failed"
                tx_hash = result.get("tx_hash", "0x" + "0" * 64)
                
                # Use zero values for a failed trade
                if trade_type == "ETH_TO_USDC":
                    amount_in = 0
                    amount_out = 0
                else:  # USDC_TO_ETH
                    amount_in = 0
                    amount_out = 0
                
                # Update portfolio with actual values 
                self.portfolio = self.get_actual_balances()
            
            # Check balances after swap
            eth_balance_after = self.portfolio["ETH"]
            usdc_balance_after = self.portfolio["USDC"]
            
            print("-" * 50)
            print(f"ETH balance: {eth_balance_after:.8f} ETH")
            print(f"USDC balance: {usdc_balance_after:.6f} USDC")
            print("-" * 50)
            
            # Get transaction summary
            print(f"Summary:")
            if trade_type == "ETH_TO_USDC":
                print(f"ETH spent: {amount_in:.8f} ETH")
                print(f"USDC received: {amount_out:.6f} USDC")
                # Calculate swap rate
                swap_rate = amount_out / amount_in if amount_in > 0 else 0
                print(f"Swap rate: 1 ETH = {swap_rate:.2f} USDC")
            else:  # USDC_TO_ETH
                print(f"USDC spent: {amount_in:.6f} USDC")
                print(f"ETH received: {amount_out:.8f} ETH")
                # Calculate swap rate
                swap_rate = amount_out / amount_in if amount_in > 0 else 0
                print(f"Swap rate: 1 USDC = {swap_rate:.8f} ETH")
            
            # Record trade in history
            trade_record = {
                "trade_type": trade_type,
                "amount_in": amount_in,
                "amount_out": amount_out,
                "price": eth_price,
                "timestamp": int(time.time()),
                "tx_hash": tx_hash,
                "status": tx_status,
                "reason": reason
            }
            
            self.trade_history.append(trade_record)
            print("-" * 50)
            return trade_record
            
        except Exception as e:
            logger.error(f"Error in execute_trade: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def display_portfolio(self):
        """Display current portfolio status"""
        # Get current market data
        market_data = self.get_market_data()
        eth_price = market_data["eth_price"]
        
        # Calculate values
        eth_value = self.portfolio["ETH"] * eth_price
        usdc_value = self.portfolio["USDC"]
        total_value = eth_value + usdc_value
        
        # Calculate percentages
        eth_pct = (eth_value / total_value) * 100 if total_value > 0 else 0
        usdc_pct = (usdc_value / total_value) * 100 if total_value > 0 else 0
        
        # Create tables
        portfolio_table = Table(title="Portfolio Summary", show_header=True, box=box.MINIMAL_HEAVY_HEAD)
        portfolio_table.add_column("Asset", style="cyan")
        portfolio_table.add_column("Balance", justify="right")
        portfolio_table.add_column("Value (USD)", justify="right")
        portfolio_table.add_column("% of Portfolio", justify="right")
        
        portfolio_table.add_row(
            "ETH", 
            f"{self.portfolio['ETH']:.9f}", 
            f"${eth_value:.2f}", 
            f"{eth_pct:.2f}%"
        )
        portfolio_table.add_row(
            "USDC", 
            f"{self.portfolio['USDC']:.2f}", 
            f"${usdc_value:.2f}", 
            f"{usdc_pct:.2f}%"
        )
        portfolio_table.add_row(
            "TOTAL", 
            "", 
            f"${total_value:.2f}", 
            f"{100:.2f}%"
        )
        
        # Display market data
        market_table = Table(title="Market Data", show_header=True, box=box.MINIMAL_HEAVY_HEAD)
        market_table.add_column("Metric", style="cyan")
        market_table.add_column("Value", justify="right")
        
        market_table.add_row("ETH Price", f"${eth_price:.2f}")
        market_table.add_row("Price Change (10m)", f"{market_data.get('price_change_pct_10m', 0.0):.2f}%")
        market_table.add_row("Trading Volume (ETH 10m)", f"{market_data.get('volume_eth_10m', 0.0):.4f} ETH")
        market_table.add_row("Trading Volume (USDC 10m)", f"${market_data.get('volume_usdc_10m', 0.0):.2f}")
        market_table.add_row("Number of Swaps (10m)", f"{market_data.get('swap_count_10m', 0)}")
        
        # Print tables
        console = Console()
        console.print(portfolio_table)
        console.print(market_table)
        
        # Print trade history
        if self.trade_history:
            # Format trade history
            trade_lines = []
            for i, trade in enumerate(self.trade_history[-5:]):  # Show last 5 trades
                timestamp = datetime.fromtimestamp(trade["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')
                if trade["trade_type"] == "ETH_TO_USDC":
                    line = f"Trade {i+1}: ETH→USDC | {trade['amount_in']:.4f} ETH → {trade['amount_out']:.2f} USDC | {timestamp}"
                else:
                    line = f"Trade {i+1}: USDC→ETH | {trade['amount_in']:.2f} USDC → {trade['amount_out']:.6f} ETH | {timestamp}"
                trade_lines.append(line)
            
            trade_panel = Panel("\n".join(trade_lines), title=f"Recent Trade History ({len(self.trade_history)} total)", border_style="green")
            console.print(trade_panel)
    
    def run(self, iterations=None, delay=TRADE_INTERVAL):
        """
        Run the trading agent for a specified number of iterations
        
        Args:
            iterations: Number of iterations (None = run indefinitely)
            delay: Delay between iterations in seconds
        """
        # Print initialized message
        console = Console()
        rpc_url = LOCAL_RPC_URL if USE_FORK else BASE_RPC_URL
        environment = "Foundry Anvil fork" if USE_FORK else "BASE mainnet"
        
        console.print(Panel(
            f"Uniswap v4 Trading Agent\nModel: {OLLAMA_MODEL}\nModel Key: {MODEL_KEY}\nEnvironment: {environment}\nEthereum node: {rpc_url}\nAccount: {self.address}\nTarget ETH Allocation: {self.target_eth_allocation:.2%}",
            title="🤖 Trading Bot Initialized",
            border_style="blue"
        ))
        
        # Display initial portfolio
        self.display_portfolio()
        
        iteration = 0
        while iterations is None or iteration < iterations:
            print(f"\nMaking trading decision at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Make trading decision
            trade = self.make_trading_decision()
            
            # Display portfolio after trade
            self.display_portfolio()
            
            # Increment counter
            iteration += 1
            
            # Break if we've reached the desired number of iterations
            if iterations is not None and iteration >= iterations:
                break
                
            # Wait before next decision
            print(f"Waiting {delay} seconds before next decision...")
            time.sleep(delay)

# For running the script directly
if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Uniswap v4 Trading Agent')
    parser.add_argument('--target-allocation', type=float, default=DEFAULT_ETH_ALLOCATION, help='Target ETH allocation (0-1, default: 0.5)')
    parser.add_argument('--iterations', type=int, default=None, help='Number of trading iterations (default: run indefinitely)')
    args = parser.parse_args()
    
    async def main():
        # Initialize trading agent with command line arguments
        agent = UniswapV4TradingAgent(
            target_eth_allocation=args.target_allocation
        )
        
        # Run the trading loop
        agent.run(iterations=args.iterations)
    
    # Run the async main function
    asyncio.run(main()) 