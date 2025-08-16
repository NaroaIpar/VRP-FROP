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

import itertools
import csv
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import main  



# Prepare CSV once before the loop
csv_path = 'results/results.csv'
write_header = not os.path.exists(csv_path)

with open(csv_path, 'a', newline='') as f:
    dict_writer = csv.DictWriter(f, fieldnames=[
        'num_vehicles', 'num_buoys', 'num_nodes', 'fixed_customers', 'load_customers', 'load_demands',
        'a_ratio', 'b_ratio', 'gamma_ratio', 'obj_lambda',
        'train_time_secs', 'train_time_human',
        'evaluate_time_secs', 'evaluate_time_human',
        'mean_reward', 'best_cost', 'best_routes', 'best_travel_cost',
        'best_expected_reward', 'best_fig_path'
    ])
    if write_header:
        dict_writer.writeheader()

    for params in itertools.product(
        [1],
        [15],
        [20],
        [True],
        [True, False],
        [True, False],
        [(0.1, 0.1, 0.8), (0.1, 0.7, 0.2), (0.6, 0.3, 0.1)],
        [0.1, 0.5, 0.7]
    ):
        num_vehicles, num_buoys, num_nodes, fixed_customers, load_customers, load_demands, ratios, obj_lambda = params
        a_ratio, b_ratio, gamma_ratio = ratios

        args = main.parse_args()
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

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

        logger = main.setup_logger(args.save_dir)
        logger.info(f"Starting SVRP-RL with {args}")

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

        policy_model = main.SVRPPolicy(
            customer_input_dim=env.customer_features_dim,
            vehicle_input_dim=env.vehicle_features_dim,
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

        train_start_time = datetime.datetime.now()
        if not args.test:
            main.train(args, env, trainer, logger)
        train_finish_time = datetime.datetime.now()

        evaluate_start_time = datetime.datetime.now()
        mean_reward, best_evaluation = main.evaluate(args, env, policy_model, args.test_size, logger)
        evaluate_finish_time = datetime.datetime.now()

        fig_path = None
        if best_evaluation.get('best_fig') is not None:
            os.makedirs('results/figs', exist_ok=True)
            current_time = datetime.datetime.now()
            timestamp_str = current_time.strftime("%Y_%m_%d_%H_%M_%S")
            fig_path = os.path.join('results/figs', f"best_{timestamp_str}.png")
            best_evaluation['best_fig'].savefig(fig_path)
            plt.close(best_evaluation['best_fig'])

        row = {
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
        }

        dict_writer.writerow(row)
        f.flush()  # ensure it's written immediately
