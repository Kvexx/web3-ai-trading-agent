# AGENTS Guidelines for This Repository

This repository contains a comprehensive Web3 AI trading agent for Uniswap V4 on BASE blockchain. When working on the project interactively with an agent (e.g. the Codex CLI) please follow the guidelines below for safe development, testing, and deployment.

⚠️ **CRITICAL WARNING**: This system involves real cryptocurrency trading. ALWAYS use test environments (Foundry fork) during development. NEVER deploy to mainnet without extensive testing and understanding of financial risks.

## 1. Use Foundry Fork for ALL Development

* **Always use Foundry fork** for testing trading strategies.
* **Never connect to mainnet** during agent development sessions.
* **Use paper trading** exclusively until thoroughly tested.
* **Top up test accounts** with fake ETH using Foundry cheats.

Start Foundry fork before any testing:
```bash
anvil --fork-url YOUR_CHAINSTACK_BASE_RPC --chain-id 8453
```

## 2. Progressive Development Workflow

Follow this progression path for safe development:

### Stage 1: Manual Testing
1. Test basic swaps with scripts
   ```bash
   python on-chain/usdc_to_eth_swap.py
   python on-chain/eth_to_usdc_swap.py
   ```

### Stage 2: Stateless Agent
2. Run stateless agent with pre-trained models
   ```bash
   python on-chain/uniswap_v4_stateless_trading_agent.py
   ```

### Stage 3: Stateful Agent
3. Test stateful agent with observation mode first
   ```bash
   python on-chain/uniswap_v4_stateful_trading_agent.py --observe-cycles 50
   ```

### Stage 4: Custom Model Training (Optional)
4. Collect data → Generate synthetic → Distill → Fine-tune → Deploy

## 3. Service Dependencies Management

Ensure all services are running before development:

| Service | Purpose | Start Command |
| ------- | ------- | ------------- |
| Foundry | Local blockchain fork | `anvil --fork-url YOUR_RPC --chain-id 8453` |
| Ollama | LLM inference | `ollama serve` |
| MLX-LM | Model training (Mac) | Installed via pip |

## 4. Configuration Management

Edit `config.py` with required settings:

```python
# CRITICAL: Use test wallet with minimal funds
PRIVATE_KEY = "YOUR_TEST_WALLET_PRIVATE_KEY"  

# Chainstack RPC endpoints
BASE_RPC_URLS = "YOUR_CHAINSTACK_ENDPOINT"

# Model selection
MODEL_KEY = "fin-r1"  # or "qwen-trader" for custom
USE_MLX_MODEL = False  # Use Ollama by default

# Trading parameters
REBALANCE_THRESHOLD = 0.5  # High to force LLM decisions
TRADE_INTERVAL = 10  # Seconds between trades
```

## 5. Data Pipeline Workflow

For custom model training, follow this sequence:

### Data Collection
```bash
# 1. Collect real swap data from BASE
python on-chain/collect_raw_data.py

# 2. Process raw data
python off-chain/process_raw_data.py
```

### Synthetic Data Generation
```bash
# 3. Train GAN (use CPU fallback on Mac if needed)
PYTORCH_ENABLE_MPS_FALLBACK=1 python off-chain/generate_synthetic_data.py train --quick-test

# 4. Generate synthetic data
python off-chain/generate_synthetic_data.py generate

# 5. Validate synthetic data quality
python off-chain/validate_synthetic_gan_data.py
```

### Model Fine-tuning
```bash
# 6. Generate teacher data (requires OpenRouter API)
python off-chain/distill_data_from_teacher.py

# 7. Prepare for MLX training
python off-chain/prepare_teacher_data_for_mlx.py

# 8. Fine-tune with LoRA
mlx_lm.lora --config off-chain/data/mlx_lm_prepared/teacher_lora_config.yaml
```

## 6. Model Deployment Options

Choose deployment strategy based on needs:

### Option A: Direct MLX Usage
```bash
mlx_lm.generate --model Qwen/Qwen2.5-3B \
  --adapter-path off-chain/models/trading_model_lora \
  --prompt "Trading prompt here" --temp 0.3
```

### Option B: Fused Model
```bash
mlx_lm.fuse --model Qwen/Qwen2.5-3B \
  --adapter-path off-chain/models/trading_model_lora \
  --save-path off-chain/models/fused_qwen
```

### Option C: Ollama Deployment
```bash
# Convert to GGUF and create Ollama model
# (See main README for detailed steps)
ollama create trader-qwen:latest -f Modelfile
```

## 7. Reinforcement Learning Enhancement

For advanced RL-enhanced models:

```bash
# Train DQN agent
python off-chain/rl_trading/train_dqn.py --timesteps 100000

# Create RL dataset
python off-chain/rl_trading/create_distillation_dataset.py

# Prepare for MLX
python off-chain/rl_trading/prepare_rl_data_for_mlx.py

# Second-stage fine-tuning
mlx_lm.lora --config off-chain/rl_trading/data/rl_data/mlx_lm_prepared/rl_lora_config.yaml
```

## 8. Testing and Validation

Always validate at each stage:

### Pool Data Verification
```bash
python on-chain/fetch_pool_data.py
python on-chain/fetch_pool_stats_from_fork.py
```

### Model Response Testing
```bash
# Test current model configuration
ollama run YOUR_MODEL "Given ETH price is $2500..."
```

### Agent Performance Monitoring
- Check transaction hashes in Foundry terminal
- Monitor portfolio balance changes
- Verify Canary words appear in custom models
- Track context usage in stateful agents

## 9. Resource Requirements

Be aware of hardware requirements:

| Component | Minimum RAM | Recommended |
| --------- | ----------- | ------------ |
| Basic Trading | 8GB | 16GB |
| GAN Training | 16GB | 32GB |
| Model Fine-tuning | 16GB | 18GB+ (Apple Silicon) |
| RL Training | 16GB | 32GB |

## 10. Common Development Tasks

### Check Account Balances
```bash
# ETH balance (Wei)
cast balance YOUR_ADDRESS --rpc-url http://localhost:8545

# USDC balance (need to convert from hex)
cast call 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913 "balanceOf(address)" YOUR_ADDRESS --rpc-url http://localhost:8545
```

### Monitor Ollama Models
```bash
ollama list
ollama show MODEL_NAME
```

### Test Model Responses
```bash
time ollama run MODEL_NAME "Trading prompt..."
```

## 11. Safety Checklist

Before any mainnet deployment:

- [ ] Extensively tested on Foundry fork
- [ ] All unit swaps verified
- [ ] Model responses validated
- [ ] Risk parameters configured
- [ ] Small test amounts only
- [ ] Stop-loss mechanisms in place
- [ ] Performance metrics tracked
- [ ] Emergency shutdown ready

## 12. Troubleshooting

### Common Issues

**"Model failed to respond"**
- Increase `TRADE_INTERVAL` in config.py
- Check Ollama is running: `curl http://localhost:11434/api/version`

**"MPS backend out of memory" (Mac)**
- Use CPU fallback: `PYTORCH_ENABLE_MPS_FALLBACK=1`
- Reduce batch size or model size

**"Transaction failed"**
- Check account has sufficient ETH for gas
- Verify USDC approval for Universal Router
- Confirm pool has adequate liquidity

**"Context window exceeded"**
- Agent will auto-summarize at 90% capacity
- Reduce observation cycles if needed
- Check `CONTEXT_WARNING_THRESHOLD` setting

## 13. Development Best Practices

* Start with observation mode to understand market behavior.
* Use smallest viable models first, scale up gradually.
* Always verify Canary words in custom models.
* Monitor gas costs even in test environments.
* Keep detailed logs of all training experiments.
* Version control model checkpoints and configs.
* Document any custom modifications clearly.

## 14. Quick Reference Commands

| Task | Command |
| ---- | ------- |
| Start Foundry fork | `anvil --fork-url YOUR_RPC --chain-id 8453` |
| Run stateless agent | `python on-chain/uniswap_v4_stateless_trading_agent.py` |
| Run stateful agent | `python on-chain/uniswap_v4_stateful_trading_agent.py` |
| Train GAN | `python off-chain/generate_synthetic_data.py train` |
| Fine-tune model | `mlx_lm.lora --config CONFIG_PATH` |
| Test model | `ollama run MODEL_NAME "prompt"` |

---

Following these practices ensures safe development, prevents loss of funds, and maintains system stability. Always prioritize testing and validation over speed of deployment. Remember: this is a learning project—treat it as such and never risk funds you cannot afford to lose.