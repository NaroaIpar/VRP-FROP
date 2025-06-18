import torch
import heapq


class InferenceStrategy:
    """Base class for inference strategies."""
    
    def __init__(self, policy_model, device='cpu'):
        """
        Args:
            policy_model: Trained policy model
            device: Device to use for tensor operations
        """
        self.policy_model = policy_model
        self.device = device
        self.policy_model.eval()  # Set to evaluation mode
        
    def solve(self, env, **kwargs):
        """
        Solve the SVRP instance.
        
        Args:
            env: SVRP environment
            
        Returns:
            solution: List of routes for each vehicle
            total_cost: Total travel cost
        """
        raise NotImplementedError("Subclasses must implement solve method")


class GreedyInference(InferenceStrategy):
    """
    Greedy inference strategy that always selects the action with highest probability.
    """
    
    # def solve(self, env):
    #     """
    #     Solve the SVRP instance using greedy action selection.
        
    #     Args:
    #         env: SVRP environment
            
    #     Returns:
    #         routes: List of routes for each vehicle
    #         total_cost: Total travel cost
    #     """
    #     # Reset environment
    #     customer_features, vehicle_features, demands = env.reset(batch_size=1)
        
    #     # Initialize hidden state
    #     hidden = None
        
    #     # Initialize routes and cost
    #     routes = [[] for _ in range(env.num_vehicles)]
    #     total_cost = 0.0
        
    #     # Track visited customers
    #     done = False
    #     step = 0
        
    #     while not done and step < 1000:  # Safety limit
    #         print(f"------------Greedy Inference Step {step}------------\n")

    #         # Forward pass through policy network
    #         with torch.no_grad():
    #             log_probs, new_hidden = self.policy_model(
    #                 customer_features, vehicle_features, demands, hidden
    #             )
    #         # print(f"Previous hidden state: {hidden}")
    #         # print(f"New hidden state: {new_hidden}")

    #         if hidden is not None:
    #             diff = 0.0
    #             if isinstance(hidden, tuple):  # LSTM case
    #                 diff = sum((h1 - h2).abs().sum().item() for h1, h2 in zip(hidden, new_hidden))
    #             else:  # GRU or single tensor
    #                 diff = (hidden - new_hidden).abs().sum().item()
                
    #             print(f"Hidden state total difference: {diff}")

    #             if diff < 1e-6:
    #                 print("[WARNING] Hidden state did not significantly change.")


            
    #         # Choose actions greedily
    #         actions = []
    #         for v in range(env.num_vehicles):
    #             # Select most probable action
    #             action = torch.argmax(log_probs[0, v]).item()
    #             actions.append(action)
                
    #             # Record route
    #             routes[v].append(action)

    #             print(f"Vehicle {v} log_probs: {log_probs[0, v]}")
    #             print(f"Vehicle {v} selected action: {action}")
    #             print(f"Vehicle {v} current route: {routes[v]}")
            
    #         # Convert to tensor
    #         actions_tensor = torch.tensor([actions], device=self.device)
            
    #         # Execute actions in environment
    #         next_customer_features, next_vehicle_features, next_demands, rewards, done_tensor = env.step(actions_tensor)
            
    #         # Update cost
    #         total_cost -= rewards.item()  # Rewards are negative costs
            
    #         # Update state
    #         customer_features = next_customer_features
    #         vehicle_features = next_vehicle_features
    #         demands = next_demands
    #         hidden = new_hidden

    #         # Debug output of customer features
    #         # print(f"Previous customer features: {customer_features}")
    #         # print(f"Next customer features: {next_customer_features}")
    #         diff = 0.0
    #         if isinstance(customer_features, tuple):  # LSTM case
    #             diff = sum((h1 - h2).abs().sum().item() for h1, h2 in zip(customer_features, next_customer_features))
    #         else:  # GRU or single tensor
    #             diff = (customer_features - next_customer_features).abs().sum().item()
    #         print(f"Customer features state total difference: {diff}")
    #         if diff < 1e-6:
    #             print("[WARNING] Customer features state did not significantly change.")

    #         # Debug output of customer features
    #         # print(f"Previous vehicle features: {vehicle_features}")
    #         # print(f"Next vehicle features: {next_vehicle_features}")
    #         diff = 0.0
    #         if isinstance(vehicle_features, tuple):  # LSTM case
    #             diff = sum((h1 - h2).abs().sum().item() for h1, h2 in zip(vehicle_features, next_vehicle_features))
    #         else:  # GRU or single tensor
    #             diff = (vehicle_features - next_vehicle_features).abs().sum().item()
    #         print(f"Vehicle features state total difference: {diff}")
    #         if diff < 1e-6:
    #             print("[WARNING] Vehicle features state did not significantly change.")

    #         # Debug output of demands
    #         # print(f"Previous demands: {demands}")
    #         # print(f"Next demands: {next_demands}")
    #         diff = 0.0
    #         if isinstance(demands, tuple):  # LSTM case
    #             diff = sum((h1 - h2).abs().sum().item() for h1, h2 in zip(demands, next_demands))
    #         else:  # GRU or single tensor
    #             diff = (demands - next_demands).abs().sum().item()
    #         print(f"Demands state total difference: {diff}")
    #         if diff < 1e-6:
    #             print("[WARNING] Demands state did not significantly change.")
            
    #         # Update step counter
    #         step += 1
            
    #         # Check if done
    #         done = done_tensor.item()
    #         print(f"--------------------------------------------------\n")

        
    #     return routes, total_cost

    
    def solve(self, env):
        """
        Solve the SVRP instance using greedy action selection.
        
        Args:
            env: SVRP environment
            
        Returns:
            routes: List of routes for each vehicle
            total_cost: Total travel cost
        """
        # Reset environment
        customer_features, vehicle_features, demands = env.reset(batch_size=1)
        
        # Initialize hidden state
        hidden = None
        
        # Initialize routes and cost
        routes = [[] for _ in range(env.num_vehicles)]
        total_cost = 0.0
        
        # Track visited customers
        done = False
        step = 0
        while not done and step < 1000:
            print(f"------------Greedy Inference Step {step}------------")

            actions = []
            if step == 0:
                # Forzar que todos los vehículos empiecen en el depot (nodo 0)
                for v in range(env.num_vehicles):
                    actions.append(0)
                    routes[v].append(0)
            else:
                # Forward pass a partir del segundo paso
                with torch.no_grad():
                    log_probs, new_hidden = self.policy_model(
                        customer_features, vehicle_features, demands, hidden
                    )

                for v in range(env.num_vehicles):
                    logits = log_probs[0, v].clone()

                    current_node = routes[v][-1]
                    mask = torch.ones_like(logits, dtype=torch.bool)

                    # Evitar repetir boyas no depot
                    for node in routes[v]:
                        if node != 0:
                            mask[node] = False

                    # No volver al depot inmediatamente después de salir
                    if current_node == 0:
                        mask[0] = False

                    # Aplicar máscara
                    logits[~mask] = -float('inf')

                    action = torch.argmax(logits).item()
                    actions.append(action)
                    routes[v].append(action)

                    print(f"Vehicle {v} log_probs: {log_probs[0, v]}")
                    print(f"Vehicle {v} logits after masking: {logits}")
                    print(f"Vehicle {v} selected action: {action}")
                    print(f"Vehicle {v} current route: {routes[v]}")

            # Ejecutar paso en el entorno
            actions_tensor = torch.tensor([actions], device=self.device)
            next_customer_features, next_vehicle_features, next_demands, rewards, done_tensor = env.step(actions_tensor)

            # Actualizar estado
            total_cost -= rewards.item()
            customer_features = next_customer_features
            vehicle_features = next_vehicle_features
            demands = next_demands
            hidden = new_hidden if step > 0 else None  # `hidden` solo se actualiza si se hizo forward

            step += 1
            done = done_tensor.item()
            print(f"--------------------------------------------------\n")

        # Indicar que todos los vehículos regresan al depot al final
        for v in range(env.num_vehicles):
            if routes[v][-1] != 0:
                routes[v].append(0)
                
        return routes, total_cost


class RandomSamplingInference(InferenceStrategy):
    """
    Random sampling inference strategy that samples multiple solutions.
    """
    
    def solve(self, env, num_samples=16):
        """
        Solve the SVRP instance by sampling multiple solutions.
        
        Args:
            env: SVRP environment
            num_samples: Number of solutions to sample
            
        Returns:
            best_routes: Best routes found
            best_cost: Cost of the best solution
        """
        best_cost = float('inf')
        best_routes = None
        
        # Sample multiple solutions
        for i in range(num_samples):
            print(f"------------Random Sampling Iteration {i}------------\n")
            # Reset environment
            customer_features, vehicle_features, demands = env.reset(batch_size=1)
            
            # Initialize hidden state
            hidden = None
            
            # Initialize routes and cost
            routes = [[] for _ in range(env.num_vehicles)]
            total_cost = 0.0
            
            # Track visited customers
            done = False
            step = 0
            
            while not done and step < 1000:  # Safety limit
                # Forward pass through policy network
                with torch.no_grad():
                    log_probs, hidden = self.policy_model(
                        customer_features, vehicle_features, demands, hidden
                    )

                print(f"------------Random Sampling Step {step}------------\n")
                
                # Sample actions
                actions = []
                for v in range(env.num_vehicles):
                    # Convert log probs to probs
                    probs = torch.exp(log_probs[0, v])
                    
                    # Sample action
                    action = torch.multinomial(probs, 1).item()
                    actions.append(action)
                    
                    # Record route
                    routes[v].append(action)
                    print(f"Vehicle {v} log_probs: {log_probs[0, v]}")
                    print(f"Vehicle {v} selected action: {action}")
                    print(f"Vehicle {v} current route: {routes[v]}")
                print(f"--------------------------------------------------\n")
                
                # Convert to tensor
                actions_tensor = torch.tensor([actions], device=self.device)
                
                # Execute actions in environment
                next_customer_features, next_vehicle_features, next_demands, rewards, done_tensor = env.step(actions_tensor)
                
                # Update cost
                total_cost -= rewards.item()  # Rewards are negative costs
                
                # Update state
                customer_features = next_customer_features
                vehicle_features = next_vehicle_features
                demands = next_demands
                
                # Update step counter
                step += 1
                
                # Check if done
                done = done_tensor.item()
            
            # Update best solution
            if total_cost < best_cost:
                best_cost = total_cost
                best_routes = routes.copy()
        
        return best_routes, best_cost


class BeamSearchInference(InferenceStrategy):
    """
    Beam search inference strategy that maintains the top-k most probable solutions.
    """
    
    def solve(self, env, beam_width=3):
        """
        Solve the SVRP instance using beam search.
        
        Args:
            env: SVRP environment
            beam_width: Number of solutions to maintain
            
        Returns:
            best_routes: Best routes found
            best_cost: Cost of the best solution
        """
        # Reset environment
        customer_features, vehicle_features, demands = env.reset(batch_size=1)
        
        # Initialize beam
        beam = [(0.0, [], [], customer_features, vehicle_features, demands, None)]
        best_complete_solution = None
        best_complete_cost = float('inf')
        
        max_steps = env.num_nodes * 2  # Heuristic for maximum steps
        
        for step in range(max_steps):
            # Expand beam
            next_beam = []
            
            for log_prob_sum, routes, all_actions, c_features, v_features, dem, hidden in beam:
                # Check if all customers are served
                if torch.all(dem[:, 1:] <= 0).item():  # All demands fulfilled
                    # Complete solution
                    if -log_prob_sum < best_complete_cost:  # Lower is better
                        best_complete_cost = -log_prob_sum
                        best_complete_solution = (routes.copy(), -log_prob_sum)
                    continue
                
                # Forward pass through policy network
                with torch.no_grad():
                    log_probs, new_hidden = self.policy_model(c_features, v_features, dem, hidden)
                
                # Generate candidates for each vehicle
                candidates = []
                
                for v in range(env.num_vehicles):
                    log_probs_v = log_probs[0, v]
                    
                    # Get top-k actions
                    values, indices = torch.topk(log_probs_v, min(beam_width, log_probs_v.size(0)))
                    
                    for i, (value, action) in enumerate(zip(values, indices)):
                        # Create new routes
                        new_routes = [r.copy() for r in routes]
                        
                        # If routes list is not long enough, extend it
                        while len(new_routes) <= v:
                            new_routes.append([])
                        
                        # Add action to route
                        new_routes[v].append(action.item())
                        
                        # Create new actions list
                        new_actions = all_actions.copy()
                        
                        # Ensure actions list has correct length
                        while len(new_actions) < step:
                            new_actions.append([0] * env.num_vehicles)
                        
                        # Create or modify actions for current step
                        if len(new_actions) <= step:
                            step_actions = [0] * env.num_vehicles
                            new_actions.append(step_actions)
                        else:
                            step_actions = new_actions[step]
                        
                        # Set action for current vehicle
                        step_actions[v] = action.item()
                        
                        # Add candidate to list
                        candidates.append((log_prob_sum + value.item(), new_routes, new_actions, value.item(), v))
                
                # Sort candidates by total log probability
                candidates.sort(key=lambda x: x[0], reverse=True)
                
                # Keep top beam_width candidates
                for cand_log_prob, cand_routes, cand_actions, action_log_prob, vehicle in candidates[:beam_width]:
                    # Execute action in environment clone
                    clone_env = self._clone_env(env, c_features, v_features, dem)
                    
                    # Create action tensor
                    action_tensor = torch.zeros(1, env.num_vehicles, dtype=torch.long, device=self.device)
                    for v, a in enumerate(cand_actions[-1]):
                        action_tensor[0, v] = a
                    
                    # Execute step
                    next_c_features, next_v_features, next_dem, rewards, done = clone_env.step(action_tensor)
                    
                    # Add to next beam
                    next_beam.append((
                        cand_log_prob,
                        cand_routes,
                        cand_actions,
                        next_c_features,
                        next_v_features,
                        next_dem,
                        new_hidden
                    ))
            
            # No candidates left
            if not next_beam:
                break
            
            # Sort and prune beam
            next_beam.sort(key=lambda x: x[0], reverse=True)
            beam = next_beam[:beam_width]
        
        # If we have a complete solution, return it
        if best_complete_solution is not None:
            routes, cost = best_complete_solution
            return routes, cost
        
        # Otherwise, return the best solution from the beam
        log_prob_sum, routes, _, _, _, _, _ = beam[0]
        return routes, -log_prob_sum
    
    def _clone_env(self, env, customer_features, vehicle_features, demands):
        """
        Create a clone of the environment in the given state.
        This is a simplified version that assumes we can just
        pass the state directly to the step function.
        
        In a real implementation, you would need to properly clone
        the entire environment state.
        
        Args:
            env: Original environment
            customer_features: Customer features tensor
            vehicle_features: Vehicle features tensor
            demands: Demands tensor
            
        Returns:
            clone_env: Cloned environment
        """
        # For simplicity, we just create a new environment object
        # In practice, you would need to properly clone the state
        from copy import deepcopy
        clone_env = deepcopy(env)
        
        # Set the environment state
        clone_env._customer_features = customer_features
        clone_env._vehicle_features = vehicle_features
        clone_env.remaining_demands = demands
        
        return clone_env