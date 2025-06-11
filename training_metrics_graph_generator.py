from main import plot_training_metrics
import argparse
from argparse import Namespace

args = Namespace(num_nodes=20, num_vehicles=1, capacity=300.0, a_ratio=0.6, b_ratio=0.2, gamma_ratio=0.2, weather_dim=3, fixed_customers=True, embedding_dim=128, epochs=100, batch_size=32, lr=0.0001, baseline_lr=0.001, entropy_weight=0.01, max_steps=100, inference='greedy', num_samples=16, beam_width=3, test_size=100, cuda=False, seed=42, save_dir='checkpoints', load_model=None, test=False, log_interval=10, save_interval=20, reoptimization=False)
rewards_history = [-62.443031311035156, -59.6675910949707, -63.193458557128906, -62.41261291503906, -63.360599517822266] * int(100/5) # Simulating 100 epochs
policy_losses = [-1570.1722412109375, -1542.272705078125, -1588.160400390625, -1553.0845947265625, -1568.27294921875]* int(100/5) # Simulating 100 epochs
baseline_losses = [23932.80078125, 22623.953125, 24088.830078125, 22996.068359375, 23530.2265625] * int(100/5) # Simulating 100 epochs

plot_training_metrics(args, rewards_history, policy_losses, baseline_losses)
