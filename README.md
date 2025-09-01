# SFROP-RL: Reinforcement Learning for Stochastic Fishing Routing Optimization Problem

This repository contains an implementation of my bachelor thesis.

## Overview

The Stochastic Fishing Routing Optimization Problem (SVRP) extends the classic Fishing Routing Problem by introducing uncertainty in customer demands and travel costs. This implementation uses a Reinforcement Learning approach to tackle this challenging optimization problem.

Key features:
- Neural network architecture with state and memory embeddings
- Attention mechanism for node selection
- REINFORCE algorithm with baseline for training
- Multiple inference strategies (greedy, random sampling, beam search)
- Support for varying levels of stochasticity in the environment

## Project Structure

```
svrp_rl/
├── main.py               # Main entry point for training/inference
├── config.py             # Configuration parameters
├── models/
│   ├── __init__.py
│   ├── attention.py      # Attention layer implementation
│   ├── embedding.py      # State and memory embedding components
│   └── policy.py         # Full policy model architecture
├── env/
│   ├── __init__.py
│   ├── svrp_env.py       # SVRP environment implementation
│   ├── weather.py        # Weather simulation for stochastic variables
│   ├── demands20.txt     # Text file to read from for the demands when num_nodes = 20
│   ├── demands50.txt     # Text file to read from for the demands when num_nodes = 50
│   ├── positions20.txt   # Text file to read from for the buoys' positons when num_nodes = 20
│   └── positions50.txt   # Text file to read from for the buoys' positons when num_nodes = 50
├── training/
│   ├── __init__.py
│   ├── reinforce.py      # REINFORCE algorithm implementation
│   └── baseline.py       # Baseline model implementation  
├── inference/
│   ├── __init__.py
│   └── inference.py      # Inference strategies (greedy, random, beam search)
```

## Installation

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

To train a model with default parameters:

```bash
python main.py
```

To customize training parameters:

```bash
python main.py --num_nodes 50 --num_vehicles 2 --embedding_dim 256 --epochs 200 --batch_size 16
python main.py --num_nodes 50 --num_vehicles 2 --num_buoys 10 --embedding_dim 256 --epochs 200 --batch_size 16 --load_customers --load_demads --obj_lambda 0.1 --inference greedy
```

### Testing

To evaluate a trained model:

```bash
python main.py --test --load_model checkpoints/model_final
```

To compare different inference strategies:

```bash
python main.py --test --load_model checkpoints/model_final --inference greedy
python main.py --test --load_model checkpoints/model_final --inference random --num_samples 32
python main.py --test --load_model checkpoints/model_final --inference beam --beam_width 5
```

## Key Parameters

- `--num_nodes`: Number of dFADs nodes plus depot
- `--num_vehicles`: Number of vessels available
- `--num_buoys`: Number of maximun buoys each vessel can visit
- `--capacity`: Maximum vessel capacity
- `--a_ratio`, `--b_ratio`, `--gamma_ratio`: Signal ratios for stochastic components
- `--embedding_dim`: Dimension of state and memory embeddings
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--inference`: Inference strategy (greedy, random, beam)

## Environment Settings

The SVRP environment models stochasticity through three components:

1. **Constant component (a_ratio)**: Fixed part of the stochastic variables
2. **Weather component (b_ratio)**: Part influenced by weather variables
3. **Noise component (gamma_ratio)**: Random noise

The model learns to leverage weather information to predict stochastic variables and make better routing decisions.