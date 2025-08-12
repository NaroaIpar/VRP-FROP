import itertools
import csv
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

# Fix import: don't use `import main.py`, use:
import main  # assuming main.py is in the same folder or PYTHONPATH

# Create results folder
os.makedirs("results", exist_ok=True)

results_table = []

# All parameter combinations


# for params in itertools.product(
#     [1, 2, 5, 10],          # num_vehicles
#     [10, 15, 30],           # num_buoys
#     [20, 50, 100],          # num_nodes
#     [True, False],          # fixed_customers
#     [True, False],          # load_customers
#     [True, False],          # load_demands
#     [(0.1, 0.1, 0.8), (0.1, 0.7, 0.2), (0.6, 0.3, 0.1)],  # ratios
#     [0.1, 0.5, 0.7]         # obj_lambda
# ):
    
for params in itertools.product(
    [1, 2, 5, 10],          # num_vehicles
    [10, 15],           # num_buoys
    [20],                   # num_nodes
    [True, False],          # fixed_customers
    [True, False],          # load_customers
    [True, False],          # load_demands
    [(0.1, 0.1, 0.8), (0.1, 0.7, 0.2), (0.6, 0.3, 0.1)],  # ratios
    [0.1, 0.5, 0.7]         # obj_lambda
):
    num_vehicles, num_buoys, num_nodes, fixed_customers, load_customers, load_demands, ratios, obj_lambda = params
    a_ratio, b_ratio, gamma_ratio = ratios

    # Prepare or override args for this combination
    args = main.parse_args()  # or replace with your args loader
    
    # Override args with combination parameters:
    args.num_vehicles = num_vehicles
    args.num_buoys = num_buoys
    args.num_nodes = num_nodes
    args.fixed_customers = fixed_customers
    args.load_customers = load_customers
    args.load_demands = load_demands
    args.a_ratio = a_ratio
    args.b_ratio = b_ratio
    args.gamma_ratio = gamma_ratio
    args.obj_lambda = obj_lambda

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    logger = main.setup_logger(args.save_dir)
    logger.info(f"Starting SVRP-RL with {args}")
    logger.info(f"Using device: {device}")

    env = main.SVRPEnvironment(
        num_nodes=args.num_nodes,
        num_vehicles=args.num_vehicles,
        num_buoys=args.num_buoys,
        capacity=args.capacity,
        weather_dim=args.weather_dim,
        a_ratio=args.a_ratio,
        b_ratio=args.b_ratio,
        gamma_ratio=args.gamma_ratio,
        device=device
    )

    if args.load_customer_positions:
        env.weather_sim.load_customer_positions = True
    if args.load_demands:
        env.weather_sim.load_demands = True

    customer_input_dim = env.customer_features_dim
    vehicle_input_dim = env.vehicle_features_dim

    logger.info(f"Customer features dimension: {customer_input_dim}")
    logger.info(f"Vehicle features dimension: {vehicle_input_dim}")

    policy_model = main.SVRPPolicy(
        customer_input_dim=customer_input_dim,
        vehicle_input_dim=vehicle_input_dim,
        embedding_dim=args.embedding_dim
    ).to(device)

    trainer = main.ReinforceTrainer(
        policy_model=policy_model,
        embedding_dim=args.embedding_dim,
        lr=args.lr,
        baseline_lr=args.baseline_lr,
        entropy_weight=args.entropy_weight,
        device=device
    )

    if args.load_model is not None:
        trainer.load_models(args.load_model)
        logger.info(f"Loaded model from {args.load_model}")


    train_start_time = datetime.datetime.now()
    # Training or testing
    if not args.test:
        # Train model
        main.train(args, env, trainer, logger)
    train_finish_time = datetime.datetime.now()


    # Evaluate model
    evaluate_start_time = datetime.datetime.now()
    mean_reward, best_evaluation = main.evaluate(args, env, policy_model, args.test_size, logger)
    evaluate_finish_time = datetime.datetime.now()
    
    logger.info(f"Final evaluation - Mean reward: {mean_reward:.4f}")

    logger.info(f"Best evaluation - Cost: {best_evaluation['best_cost']:.4f}, "
                f"Routes: {best_evaluation['best_routes']}, "
                f"Travel cost: {best_evaluation['best_travel_cost']}, "
                f"Expected reward: {best_evaluation['best_expected_reward']}")

    # ensure best_fig saved and store its path instead of object
    fig_path = None
    if best_evaluation.get('best_fig') is not None:
        fig = best_evaluation['best_fig']
        os.makedirs('results/figs', exist_ok=True)
        fig_path = os.path.join('results/figs', f"best_{int(time.time())}.png")
        fig.savefig(fig_path)
        plt.close(fig)
    
    
    
    
    
    # Save results for this combination
    results_table.append({
        'num_vehicles': num_vehicles,
        'num_buoys': num_buoys,
        'num_nodes': num_nodes,
        'fixed_customers': fixed_customers,
        'load_customers': load_customers,
        'load_demands': load_demands,
        'a_ratio': a_ratio,
        'b_ratio': b_ratio,
        'gamma_ratio': gamma_ratio,
        'obj_lambda': obj_lambda,
        'train_time_secs': (train_finish_time - train_start_time).total_seconds(),
        'train_time_human': str(train_finish_time - train_start_time),
        'evaluate_time_secs': (evaluate_finish_time - evaluate_start_time).total_seconds(),
        'evaluate_time_human': str(evaluate_finish_time - evaluate_start_time),
        'mean_reward': float(mean_reward),
        'best_cost': float(best_evaluation['best_cost']),
        'best_routes': str(best_evaluation['best_routes']),
        'best_travel_cost': float(best_evaluation['best_travel_cost']),
        'best_expected_reward': float(best_evaluation['best_expected_reward']),
        'best_fig_path': fig_path
    })

# After all runs, save to CSV
keys = results_table[0].keys()
with open('results/results.csv', 'w', newline='') as f:
    dict_writer = csv.DictWriter(f, keys)
    dict_writer.writeheader()
    dict_writer.writerows(results_table)
