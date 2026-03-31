# Contributing to web3-ai-trading-agent

Thank you for your interest in contributing to the web3-ai-trading-agent project! This document provides guidelines and instructions for contributing to this Web3 AI trading bot for Uniswap V4 on Base blockchain.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Project Architecture](#project-architecture)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Security Considerations](#security-considerations)
- [Contribution Areas](#contribution-areas)
- [Submitting Changes](#submitting-changes)

## Development Environment Setup

### Prerequisites

- **Python 3.9+** - Core language for the trading agent
- **Foundry/Anvil** - Local blockchain fork for testing
- **Ollama** - LLM inference engine
- **Git** - Version control

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kvexx/web3-ai-trading-agent.git
   cd web3-ai-trading-agent
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Ollama** (for LLM inference):
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Start Ollama server
   ollama serve
   ```

4. **Install Foundry** (for local blockchain testing):
   ```bash
   curl -L https://foundry.paradigm.xyz | bash
   foundryup
   ```

5. **Configure environment:**
   ```bash
   # Copy and edit config.py with your settings
   # Set USE_FORK = True for safe testing
   ```

### Running the Development Environment

1. **Start Foundry Anvil fork:**
   ```bash
   anvil --fork-url YOUR_BASE_RPC_URL --chain-id 8453
   ```

2. **Start Ollama server:**
   ```bash
   ollama serve
   ```

3. **Test basic functionality:**
   ```bash
   python on-chain/usdc_to_eth_swap.py
   ```

## Project Architecture

### Directory Structure

```
web3-ai-trading-agent/
├── on-chain/                    # On-chain interaction scripts
│   ├── usdc_to_eth_swap.py      # USDC → ETH swap implementation
│   ├── eth_to_usdc_swap.py      # ETH → USDC swap implementation
│   ├── uniswap_v4_swapper.py    # Core Uniswap V4 swap logic
│   ├── uniswap_v4_stateless_trading_agent.py  # Stateless AI agent
│   ├── uniswap_v4_stateful_trading_agent.py   # Stateful AI agent
│   ├── fetch_pool_data.py       # Pool data retrieval
│   └── *.abi                    # Contract ABIs
├── off-chain/                   # Off-chain ML components
│   ├── models/                  # Trained model storage
│   ├── gan/                     # GAN for synthetic data
│   ├── rl_trading/              # Reinforcement learning
│   ├── collect_raw_data.py      # Data collection
│   ├── process_raw_data.py      # Data preprocessing
│   ├── generate_synthetic_data.py
│   └── distill_data_from_teacher.py
├── config.py                    # Configuration settings
└── requirements.txt             # Python dependencies
```

### Core Components

1. **Uniswap V4 Integration**
   - Pool Manager: `0x498581fF718922c3f8e6A244956aF099B2652b2b`
   - Position Manager: `0x7C5f5A4bBd8fD63184577525326123B519429bDc`
   - Universal Router: `0x6ff5693b99212da76ad316178a184ab56d299b43`

2. **Trading Agents**
   - **Stateless Agent**: Uses pre-trained models without memory
   - **Stateful Agent**: Maintains trading state and learns from observations

3. **ML Pipeline**
   - Data collection from Base blockchain
   - Synthetic data generation via GAN
   - Model training and distillation
   - Reinforcement learning optimization

## Coding Standards

### Python Style Guide

- Follow **PEP 8** style guidelines
- Use **type hints** for function parameters and return values
- Maximum line length: **100 characters**
- Use **docstrings** for all public functions and classes

```python
def fetch_pool_stats(pool_address: str, block_number: int) -> dict:
    """
    Fetch pool statistics for a given Uniswap V4 pool.
    
    Args:
        pool_address: The pool contract address
        block_number: The block number to query at
        
    Returns:
        Dictionary containing pool statistics
    """
    # Implementation
```

### Code Organization

- Keep functions focused and single-purpose
- Use meaningful variable names
- Add comments for complex logic
- Separate configuration from business logic

### Error Handling

```python
try:
    result = execute_swap(amount_in, token_in, token_out)
except InsufficientLiquidity as e:
    logger.error(f"Insufficient liquidity: {e}")
    return None
except Exception as e:
    logger.exception("Unexpected error during swap")
    raise
```

## Testing Guidelines

### Testing Philosophy

**⚠️ CRITICAL**: Always use Foundry fork for testing. Never test with real funds.

### Test Levels

1. **Unit Tests** - Test individual functions
2. **Integration Tests** - Test swap execution on fork
3. **Agent Tests** - Test trading strategies in simulation

### Running Tests

```bash
# Start local fork
anvil --fork-url YOUR_BASE_RPC_URL --chain-id 8453

# Test basic swaps
python on-chain/usdc_to_eth_swap.py
python on-chain/eth_to_usdc_swap.py

# Test stateless agent
python on-chain/uniswap_v4_stateless_trading_agent.py

# Test stateful agent in observation mode
python on-chain/uniswap_v4_stateful_trading_agent.py --observe-cycles 50
```

### Test Data

- Use historical block data for consistent testing
- Document expected behavior for each test case
- Include edge cases (zero amounts, extreme prices)

## Security Considerations

### ⚠️ Financial Risk Warning

This system involves real cryptocurrency trading. Follow these rules:

1. **Never commit private keys** to the repository
2. **Always use test environments** during development
3. **Set `USE_FORK = True`** in config.py for all testing
4. **Review all transactions** before signing
5. **Test thoroughly** before mainnet deployment

### Secure Development Practices

- Use environment variables for sensitive data
- Validate all inputs before processing
- Implement proper error handling
- Log all trading decisions for audit

### Smart Contract Interactions

- Verify contract addresses against official Uniswap V4 deployments
- Check token decimals (USDC = 6, ETH/WETH = 18)
- Implement slippage protection
- Handle reverts gracefully

## Contribution Areas

### High Priority

1. **Enhanced Trading Strategies**
   - Implement momentum-based strategies
   - Add mean reversion algorithms
   - Develop arbitrage detection

2. **Risk Management**
   - Position sizing algorithms
   - Stop-loss implementation
   - Portfolio diversification

3. **Model Improvements**
   - Better feature engineering
   - Ensemble model approaches
   - Online learning capabilities

### Medium Priority

1. **Data Pipeline**
   - Real-time data streaming
   - Additional data sources
   - Data validation and cleaning

2. **Monitoring & Analytics**
   - Performance dashboards
   - Trade history analysis
   - Profit/loss tracking

3. **Documentation**
   - API documentation
   - Strategy explanations
   - Tutorial videos

### Good First Issues

- Add type hints to existing functions
- Improve error messages
- Add logging statements
- Write unit tests for utility functions
- Document configuration options

## Submitting Changes

### Pull Request Process

1. **Fork the repository** and create your branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards

3. **Test your changes** using Foundry fork

4. **Update documentation** if needed

5. **Commit your changes**:
   ```bash
   git commit -m "feat: Add feature description"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** with:
   - Clear description of changes
   - Testing instructions
   - Any breaking changes noted

### Commit Message Format

Follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

Example:
```
feat: Add stop-loss mechanism to stateful agent

- Implement trailing stop-loss logic
- Add configuration for stop-loss percentage
- Include tests for stop-loss triggers
```

## Questions?

- Check existing [AGENTS.md](./AGENTS.md) for development guidelines
- Review the README for setup instructions
- Open an issue for discussion before major changes

Thank you for contributing to web3-ai-trading-agent! 🚀