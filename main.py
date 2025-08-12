import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import logging
from tqdm import tqdm

import matplotlib.gridspec as gridspec
import textwrap

from models.policy import SVRPPolicy
from env.svrp_env import SVRPEnvironment
from training.reinforce import ReinforceTrainer
from inference.inference import GreedyInference, RandomSamplingInference, BeamSearchInference


def parse_args():
    parser = argparse.ArgumentParser(description='SVRP-RL')
    
    # Environment settings
    parser.add_argument('--num_nodes', type=int, default=20, help='Number of nodes (customers + depot)')
    parser.add_argument('--num_vehicles', type=int, default=1, help='Number of vehicles')
    parser.add_argument('--capacity', type=float, default=50000.0, help='Vehicle capacity')
    parser.add_argument('--a_ratio', type=float, default=0.6, help='Constant component ratio')
    parser.add_argument('--b_ratio', type=float, default=0.2, help='Weather component ratio')
    parser.add_argument('--gamma_ratio', type=float, default=0.2, help='Noise component ratio')
    parser.add_argument('--weather_dim', type=int, default=3, help='Weather dimension')
    parser.add_argument('--fixed_customers', action='store_true', help='Use fixed customer positions')
    parser.add_argument('--num_buoys', type=int, default=1, help='Number of buoys to visit')
    parser.add_argument('--obj_lambda', type=float, default=0.5, help='Objective function weight for costs and delivery')
    
    parser.add_argument('--load_customer_positions', action='store_true', help='Load fixed customer positions from file')
    parser.add_argument('--load_demands', action='store_true', help='Load fixed customer positions from file')

    
    # Model settings
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--baseline_lr', type=float, default=1e-3, help='Baseline learning rate')
    parser.add_argument('--entropy_weight', type=float, default=0.01, help='Entropy regularization weight')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps per episode')
    
    # Inference settings
    parser.add_argument('--inference', type=str, default='greedy', choices=['greedy', 'random', 'beam'], help='Inference strategy')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples for random sampling')
    parser.add_argument('--beam_width', type=int, default=3, help='Beam width for beam search')
    parser.add_argument('--test_size', type=int, default=100, help='Number of test instances')
    
    # Other settings
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save models')
    parser.add_argument('--load_model', type=str, default=None, help='Path to load model from')
    parser.add_argument('--test', action='store_true', help='Test mode (no training)')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval for training')
    parser.add_argument('--save_interval', type=int, default=20, help='Save interval for models')
    parser.add_argument('--reoptimization', action='store_true', help='Use reoptimization strategy')

    
    return parser.parse_args()


def setup_logger(save_dir):
    """Set up the logger for training and testing."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    log_file = os.path.join(save_dir, 'svrp_log.txt')
    
    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger()

def visualize_route(env, routes, title=None, save_path=None, save = True):
    """
    Visualize routes with customer layout and written route text.
    Auto-saves image with incrementing ID in {args.save_dir}/{args.inference}/.

    Args:
        env: SVRP environment
        routes: List of routes per vehicle
        title: Plot title
        save_path: Expected base path like args.save_dir/route_beam.png
    """
    print("Visualizing route...")
    print(f"env: {env}")
    print(f"routes: {routes}")
    print(f"title: {title}")
    print(f"save_path: {save_path}")

    # Rewrite save_path to desired directory and filename format
    if save_path:
        base_dir = os.path.dirname(save_path)
        file_root = os.path.basename(save_path)
        name_part = os.path.splitext(file_root)[0]  # "route_beam"

        # Extract "beam" from "route_beam"
        if "_" in name_part:
            inference = name_part.split("_")[1]
        else:
            inference = "default"

        # New target folder: save_dir/routes_imgs/inference/
        root_dir = os.path.dirname(base_dir)  # -> args.save_dir
        new_dir = os.path.join(root_dir, "routes_imgs", inference)
        os.makedirs(new_dir, exist_ok=True)

        # New base filename: route_inference_N.png
        base_name = f"route_{inference}"
        ext = ".png"
        count = 0
        final_name = f"{base_name}_{count}{ext}"
        final_path = os.path.join(new_dir, final_name)
        while os.path.exists(final_path):
            count += 1
            final_name = f"{base_name}_{count}{ext}"
            final_path = os.path.join(new_dir, final_name)
    else:
        final_path = None

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 2], wspace=0.3)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Left: route text
    ax_text = plt.subplot(gs[0])
    ax_text.axis('off')
    route_text = "\n\n".join([
        f"Vehicle {i}:\n" + textwrap.fill(str(route), width=55)
        for i, route in enumerate(routes)
    ])
    ax_text.text(0, 1, route_text, va='top', ha='left', fontsize=10, transform=ax_text.transAxes)

    # Right: route plot
    ax_plot = plt.subplot(gs[1])
    if hasattr(env.weather_sim, 'fixed_customer_positions') and env.weather_sim.fixed_customer_positions is not None:
        customer_positions = env.weather_sim.fixed_customer_positions[0].cpu().numpy()
    else:
        num_nodes = env.num_nodes
        customer_positions = np.zeros((num_nodes, 2))
        customer_positions[0] = [0.5, 0.5]
        for i in range(1, num_nodes):
            customer_positions[i] = np.random.rand(2)
        print("Warning: Using random positions for visualization")

    for idx in range(1, len(customer_positions)):
        x, y = customer_positions[idx]
        ax_plot.scatter(x, y, c='blue', s=50)
        # ax_plot.text(x, y + 0.01, str(idx), fontsize=9, ha='center', va='bottom')
        demand = env.demands[0][idx].item() 
        ax_plot.text(x, y + 0.01, f"{idx} ({demand:.1f})", fontsize=9, ha='center', va='bottom')


    x0, y0 = customer_positions[0]
    ax_plot.scatter(x0, y0, c='red', s=100, marker='*', label='Depot')
    ax_plot.text(x0, y0 + 0.01, '0', fontsize=10, fontweight='bold', ha='center', va='bottom')

    colors = ['green', 'orange', 'purple', 'cyan', 'magenta']
    for i, route in enumerate(routes):
        route_positions = customer_positions[route]
        ax_plot.plot(route_positions[:, 0], route_positions[:, 1],
                     label=f'Vehicle {i}', color=colors[i % len(colors)], linewidth=2)

    if title:
        ax_plot.set_title(title)
    ax_plot.legend()
    plt.tight_layout()

    if final_path and save:
        plt.savefig(final_path)
        print(f"Saved route image to: {final_path}")
    else:
        # show only if user asked to save but no path provided
        if (not final_path) and save:
            plt.show()
        # if save is False -> do nothing (headless)
        
    plt.close()
    return fig


def plot_training_metrics(args, rewards_history, policy_losses, baseline_losses):
    """
    Plots training metrics and saves the plot with unique name in training_metrics_imgs/<inference>/.
    """
    print("Plotting training metrics...")
    print(f"args.save_dir: {args.save_dir}")
    print(f"args.inference: {args.inference}")
    print(f"args: {args}")
    print(f"rewards_history: {rewards_history[:5]}... (total {len(rewards_history)})")
    print(f"policy_losses: {policy_losses[:5]}... (total {len(policy_losses)})")
    print(f"baseline_losses: {baseline_losses[:5]}... (total {len(baseline_losses)})")

    # Root: checkpoints/training_metrics_imgs/beam
    save_dir = os.path.join(args.save_dir, "training_metrics_imgs", args.inference)
    os.makedirs(save_dir, exist_ok=True)

    # Find next available filename: training_metrics_0.png, _1.png, ...
    base_name = "training_metrics"
    ext = ".png"
    count = 0
    filename = f"{base_name}_{count}{ext}"
    filepath = os.path.join(save_dir, filename)
    while os.path.exists(filepath):
        count += 1
        filename = f"{base_name}_{count}{ext}"
        filepath = os.path.join(save_dir, filename)


    # Plot
    plt.figure(figsize=(16, 5))

    plt.subplot(1, 4, 1)
    plt.axis('off')
    args_text = "\n".join([f"{k}: {v}" for k, v in vars(args).items()])
    plt.text(0, 1, args_text, va='top', ha='left', fontsize=9, wrap=True)

    plt.subplot(1, 4, 2)
    plt.plot(rewards_history)
    plt.title('Rewards')
    plt.xlabel('Episode')

    plt.subplot(1, 4, 3)
    plt.plot(policy_losses)
    plt.title('Policy Losses')
    plt.xlabel('Episode')

    plt.subplot(1, 4, 4)
    plt.plot(baseline_losses)
    plt.title('Baseline Losses')
    plt.xlabel('Episode')

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def train(args, env, trainer, logger):
    """
    Train the model.
    
    Args:
        args: Command line arguments
        env: SVRP environment
        trainer: REINFORCE trainer
        logger: Logger
    """
    logger.info(f"Starting training for {args.epochs} epochs")
    
    # Track metrics
    rewards_history = []
    policy_losses = []
    baseline_losses = []
    
    # Training loop
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train for one episode
        reward, policy_loss, baseline_loss = trainer.train_episode(
            env=env,
            batch_size=args.batch_size,
            max_steps=args.max_steps
        )
        
        # Track metrics
        rewards_history.append(reward)
        policy_losses.append(policy_loss)
        baseline_losses.append(baseline_loss)
        
        # Log progress
        if (epoch + 1) % args.log_interval == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs} | "
                        f"Reward: {reward:.4f} | "
                        f"Policy Loss: {policy_loss:.4f} | "
                        f"Baseline Loss: {baseline_loss:.4f} | "
                        f"Time: {time.time() - start_time:.2f}s")
        
        # Save model
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f"model_epoch_{epoch+1}")
            trainer.save_models(save_path)
            logger.info(f"Saved model to {save_path}")
            
            # Evaluate on test set
            test_reward, _ = evaluate(args, env, trainer.policy_model, 10, logger, save_figs=False)
            logger.info(f"Test reward: {test_reward:.4f}")
    
    # Save final model
    save_path = os.path.join(args.save_dir, "model_final")
    trainer.save_models(save_path)
    logger.info(f"Saved final model to {save_path}")

    plot_training_metrics(args, rewards_history, policy_losses, baseline_losses)
    
    logger.info("Training completed")


def evaluate(args, env, policy_model, num_instances=100, logger=None, save_figs=True):
    """
    Evaluate the model on multiple instances.
    """
    total_reward = 0.0
    best_cost = float('inf')
    best_routes = None

    best_travel_cost = None
    best_expected_reward = None
    best_fig = None   # ensure defined even if we don't save

    # Create inference strategy
    if args.inference == 'greedy':
        inference_strategy = GreedyInference(policy_model, device=next(policy_model.parameters()).device)
    elif args.inference == 'random':
        inference_strategy = RandomSamplingInference(policy_model, device=next(policy_model.parameters()).device)
    else:  # beam
        inference_strategy = BeamSearchInference(policy_model, device=next(policy_model.parameters()).device)

    # Evaluate on multiple instances
    for i in tqdm(range(num_instances), desc="Evaluating"):

        # Solve instance with appropriate parameters for each strategy
        if args.inference == 'random':
            routes, cost = inference_strategy.solve(env=env, num_samples=args.num_samples)
        elif args.inference == 'beam':
            routes, cost = inference_strategy.solve(env=env, beam_width=args.beam_width)
        else:  # greedy
            routes, cost, total_travel_costs, total_expected_reward = inference_strategy.solve(env=env)

        total_reward -= cost  # Reward is negative cost

        # Track best route
        if cost < best_cost:
            best_cost = cost
            best_routes = routes
            best_travel_cost = total_travel_costs
            best_expected_reward = total_expected_reward
                
        # Log instance results
        if logger and i < 5:  # Only log first 5 instances
            # logger.info(f"Instance {i+1}: Cost = {cost:.4f}, Routes = {routes}")
            logger.info(f"Instance {i+1}: Cost = {cost:.4f}, Travel cost contribution {total_travel_costs:.4f}, Expected reward contribution = {total_expected_reward:.4f}, Routes = {routes}")
            
            # Visualize route for first instance
            if save_figs and i == 0:
                visualize_route(
                    env=env,
                    routes=routes,
                    title=f"Routes (Objective function value: {cost:.4f})(travel_cost: {total_travel_costs:.4f}, expected reward: {total_expected_reward:.4f})",
                    save_path=os.path.join(args.save_dir, f"route_{args.inference}.png")
                )

    # Track best route
    # Log instance results
    if  best_routes is not None:
        if logger:
            logger.info(f"BEST (Objective function value: {best_cost:.4f})(travel_cost: {best_travel_cost:.4f}, expected reward: {best_expected_reward:.4f})")
        
        if save_figs:
            # Visualize route
            best_fig = visualize_route(
                env=env,
                routes=best_routes,
                title=f"BEST (Objective function value: {best_cost:.4f})(travel_cost: {best_travel_cost:.4f}, expected reward: {best_expected_reward:.4f})",
                save_path=os.path.join(args.save_dir, f"route_{args.inference}.png")
            )
                
    
    # Calculate mean reward
    mean_reward = total_reward / num_instances
    
    if logger:
        logger.info(f"Evaluation completed. Mean reward: {mean_reward:.4f}")

    best_evaluation = {
        'best_cost': best_cost,
        'best_routes': best_routes,
        'best_travel_cost': best_travel_cost,
        'best_expected_reward': best_expected_reward,
        'best_fig': best_fig
    }
    
    return mean_reward, best_evaluation


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine device
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    
    # Set up logger
    logger = setup_logger(args.save_dir)
    logger.info(f"Starting SVRP-RL with {args}")
    logger.info(f"Using device: {device}")
    
    # Create environment
    env = SVRPEnvironment(
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
    
    # Determine input dimensions
    customer_input_dim = env.customer_features_dim
    vehicle_input_dim = env.vehicle_features_dim
    
    logger.info(f"Customer features dimension: {customer_input_dim}")
    logger.info(f"Vehicle features dimension: {vehicle_input_dim}")
    
    # Create policy model
    policy_model = SVRPPolicy(
        customer_input_dim=customer_input_dim,
        vehicle_input_dim=vehicle_input_dim,
        embedding_dim=args.embedding_dim
    ).to(device)
    
    # Create trainer
    trainer = ReinforceTrainer(
        policy_model=policy_model,
        embedding_dim=args.embedding_dim,
        lr=args.lr,
        baseline_lr=args.baseline_lr,
        entropy_weight=args.entropy_weight,
        device=device
    )
    
    # Load model if specified
    if args.load_model is not None:
        trainer.load_models(args.load_model)
        logger.info(f"Loaded model from {args.load_model}")
    
    # Training or testing
    if not args.test:
        # Train model
        train(args, env, trainer, logger)

    # Evaluate model
    logger.info(f"Evaluating model with {args.inference} inference strategy")
    mean_reward, best_evaluation = evaluate(args, env, policy_model, args.test_size, logger)
    logger.info(f"Final evaluation - Mean reward: {mean_reward:.4f}")

    logger.info(f"Best evaluation - Cost: {best_evaluation['best_cost']:.4f}, "
                f"Routes: {best_evaluation['best_routes']}, "
                f"Travel cost: {best_evaluation['best_travel_cost']}, "
                f"Expected reward: {best_evaluation['best_expected_reward']}")


if __name__ == "__main__":
    main()