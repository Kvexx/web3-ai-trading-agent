import os

# RPC URLs
BASE_RPC_URLS = [
    "YOUR_CHAINSTACK_NODE_ENDPOINT",
    "YOUR_CHAINSTACK_NODE_ENDPOINT",
]
BASE_RPC_URL = BASE_RPC_URLS[0]  # Default RPC for backward compatibility

# Model configuration - merging settings from both files
OLLAMA_MODEL = os.environ.get("TRADING_MODEL", "qwen2.5:3b")

LOCAL_RPC_URL = "http://localhost:8545"

# Trading Environment Configuration
# Set to True for Foundry Anvil fork (safe testing), False for real BASE mainnet (real money!)
USE_FORK = True  # ⚠️ DANGER: Set to False only when you want to trade with REAL funds

# Uniswap V4 contracts on BASE
# https://docs.uniswap.org/contracts/v4/deployments
UNISWAP_V4_POOL_MANAGER = "0x498581fF718922c3f8e6A244956aF099B2652b2b"
UNISWAP_V4_POSITION_DESCRIPTOR = "0x25D093633990DC94BeDEeD76C8F3CDaa75f3E7D5"
UNISWAP_V4_POSITION_MANAGER = "0x7C5f5A4bBd8fD63184577525326123B519429bDc"
UNISWAP_V4_QUOTER = "0x0d5e0F971ED27FBfF6c2837bf31316121532048D"
UNISWAP_V4_STATE_VIEW = "0xA3c0c9b65baD0b08107Aa264b0f3dB444b867A71"
UNISWAP_V4_UNIVERSAL_ROUTER = "0x6ff5693b99212da76ad316178a184ab56d299b43"
UNISWAP_V4_ETH_USDC_POOL = "0x96d4b53a38337a5733179751781178a2613306063c511b78cd02684739288c0a" # https://app.uniswap.org/explore/pools/base/0x96d4b53a38337a5733179751781178a2613306063c511b78cd02684739288c0a

# Token addresses
BASE_USDC_ADDRESS = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"  # 6 decimals — pay attention to 6 decimals as most tokens are 18 decimals
ETH_ADDRESS = "0x0000000000000000000000000000000000000000"  # ETH native token address with standard 18 decimals
WETH_ADDRESS = "0x4200000000000000000000000000000000000006"  # WETH address on BASE with standard 18 decimals

# Block times (in seconds)
BASE_BLOCK_TIME = 2  # BASE chain block time is ~2 seconds

# Data collection parameters
START_BLOCK = 25350999 # This is the block number in which the Uniswap V4: Universal Router contract was deployed. Here's the transaction: https://basescan.org/tx/0x0efe6f4f59683fd326dcefe5c07f7b072740ae02fcbe81dbc1755e4aba5fe1f2
BATCH_SIZE = 200

# Model configuration - supports Ollama, MLX, and OpenRouter
# For Ollama context capacity, run `ollama show MODEL_NAME`
# You can grab Ollama-ready models from https://ollama.com/library
# The models listed are examples, adjust to what you have locally
AVAILABLE_MODELS = {
    'qwen3b': {
        'model': 'qwen2.5:3b',
        'context_capacity': 32768,
        'provider': 'ollama'
    },
    'qwen-trader': {  
        'model': 'trader-qwen:latest',
        'context_capacity': 32768,
        'provider': 'ollama'
    },
    'fin-r1': { # https://huggingface.co/Mungert/Fin-R1-GGUF
        'model': 'hf.co/Mungert/Fin-R1-GGUF',
        'context_capacity': 32768,
        'provider': 'ollama'
    },
    'grok4': {
        'model': 'x-ai/grok-4', # See https://openrouter.ai/x-ai/grok-4
        'context_capacity': 256000,  # See https://openrouter.ai/x-ai/grok-4
        'provider': 'openrouter'
    },
    'kimi-k2': {
        'model': 'moonshotai/kimi-k2:free', # See https://openrouter.ai/moonshotai/kimi-k2:free
        'context_capacity': 65536,  # See https://openrouter.ai/moonshotai/kimi-k2:free
        'provider': 'openrouter'
    },
}

# Model key from AVAILABLE_MODELS to be used with the agent
# Use "grok4" for OpenRouter's Grok-4, "qwen3b" for local Ollama, etc.
MODEL_KEY = "kimi-k2"

OLLAMA_MODEL = AVAILABLE_MODELS[MODEL_KEY]['model']
OLLAMA_URL = "http://localhost:11434"

# Model selection flag - set to False to use Ollama instead of MLX
USE_MLX_MODEL = False

# MLX model configuration
MLX_BASE_MODEL = "Qwen/Qwen2.5-3B"  # Base Qwen model
MLX_ADAPTER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "off-chain", "models", "trading_model_lora")  # Path to the LoRA adapter
MLX_TEMPERATURE = 0.0  # Temperature setting for model inference

# Trading parameters
TRADE_INTERVAL = 10  # 10 seconds between trade decisions; also make sure your MODEL_KEY can handle the response window

# Private key for your trading account
PRIVATE_KEY = "YOUR_PRIVATE_KEY"

# Strategy parameters
MIN_STRATEGY_DURATION = 30 * 60  # 30 minutes minimum
MAX_STRATEGY_DURATION = 24 * 60 * 60  # 24 hours maximum
REBALANCE_THRESHOLD = 0.5  # 50% deviation -- practically disabling rebalancing and making it a an autonomous LLM agent
DEFAULT_ETH_ALLOCATION = 0.5  # Default target allocation of 50% ETH / 50% USDC

# LLM context parameters
# Context window management settings
CONTEXT_WARNING_THRESHOLD = 0.9  # 90% of context capacity
SUMMARIZATION_COOLDOWN = 2 * 60  # 2 minutes between summarizations

# Get context capacity from model configuration
def get_context_capacity(model_key=MODEL_KEY, test_mode=False):
    """Get the appropriate context capacity based on model key and test mode"""
    # If in test mode, use a smaller context to test summarization
    if test_mode:
        return 2000
    
    # Get context capacity for the specified model_key
    model_info = AVAILABLE_MODELS.get(model_key.lower())
    if not model_info:
        raise ValueError(f"Invalid model key: {model_key}. Must be one of: {', '.join(AVAILABLE_MODELS.keys())}")
    
    return int(model_info['context_capacity'])

# GAN Configuration
GAN_CONFIG = {
    # Model architecture
    'sequence_length': 24,  # Keep this consistent
    'condition_window': 20, # Keep this consistent
    'batch_size': 32,       # Smaller batches for more gradient updates
    'latent_dim': 48,       # Latent space
    'hidden_dim': 256,      # Must be divisible by num_heads (256 = 8 * 32)
    'num_heads': 8,         # Number of attention heads for Transformer
    'num_layers': 4,        # Number of Transformer decoder layers
    'dropout': 0.1,         # Dropout for Transformer layers
    
    # Training parameters
    'epochs': 150,          # Early stoppage below can affect this
    'learning_rate': 0.0002, 
    'n_critic': 5,          # Keep this consistent
    'lambda_gp': 10.0,      # Keep this consistent
    'early_stopping_patience': 30, # More patience to find better minima
    'seed': 42,
    
    # File paths
    'output_dir': 'off-chain/models/gan_uniswap_v4',
    'checkpoint_dir': 'off-chain/checkpoints/gan_uniswap_v4',
    'processed_data_path': 'off-chain/data/processed/processed_swaps.csv',
    'synthetic_data_dir': 'off-chain/data/synthetic'
}

# Quick test configuration with smaller model and fewer epochs
QUICK_TEST_GAN_CONFIG = {
    # Model architecture — smaller and faster
    'sequence_length': 24,  # Keep this consistent 
    'condition_window': 20, # Keep this consistent
    'batch_size': 64,       # Larger batch size for faster iteration
    'latent_dim': 16,       # Smaller latent space for faster training
    'hidden_dim': 64,       # Must be divisible by num_heads (64 = 8 * 8)
    'num_heads': 8,         # Number of attention heads
    'num_layers': 2,        # Fewer layers for quick testing
    'dropout': 0.1,         # Dropout for Transformer layers
    
    # Training parameters — faster
    'epochs': 20,           # Just 20 epochs for quick testing
    'learning_rate': 0.0002,
    'n_critic': 3,          # Fewer critic steps
    'lambda_gp': 10.0,
    'early_stopping_patience': 5, # Quicker early stopping
    'seed': 42,
    
    # File paths -- use different paths to avoid overwriting production models
    'output_dir': 'off-chain/models/gan_uniswap_v4_test',
    'checkpoint_dir': 'off-chain/checkpoints/gan_uniswap_v4_test',
    'processed_data_path': 'off-chain/data/processed/processed_swaps.csv',
    'synthetic_data_dir': 'off-chain/data/synthetic_test'
}

# Global quick test mode flag
QUICK_TEST_MODE = False  # Set to True to enable quick test mode by default

OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY"