import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineModel(nn.Module):
    """
    Baseline model for estimating the expected return of a state.
    Used to reduce variance in the REINFORCE algorithm.
    
    As described in the paper: b_φ(I^t_s, h^t_k) is trained to minimize
    L(φ) = (1/S)∑^S_s ∑^T_s_t=1 ||b_φ(I^t_s, h^t_k) - C(I^t_s, h^t_k)||²
    """
    
    def __init__(self, embedding_dim):
        """
        Args:
            embedding_dim: Dimension of state embeddings
        """
        super(BaselineModel, self).__init__()
        
        # Define layers
        # input_dim = 16  # num_nodes = 10, input_dim = 14 + 2 = 16
        input_dim = 26 # num_nodes = 20, input_dim = 24 + 2 = 26
        # input_dim = 56 # num_nodes = 50, input_dim = 54 + 2 = 56
        self.fc1 = nn.Linear(input_dim, embedding_dim)
        # print(f"self.fc1.weight shape before adaptation: {self.fc1.weight.shape}")
        self.fc2 = nn.Linear(embedding_dim, embedding_dim // 2)
        self.fc3 = nn.Linear(embedding_dim // 2, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, customer_features, vehicle_features):
        """
        Estimate the expected return for the given state.
        
        Args:
            customer_features: Tensor of shape [batch_size, num_nodes, feature_dim]
            vehicle_features: Tensor of shape [batch_size, num_vehicles, feature_dim]
            
        Returns:
            Tensor of shape [batch_size, 1] containing estimated returns
        """
        batch_size = customer_features.size(0)
        customer_feautre_dim = customer_features.size(2)
        vehicle_feature_dim = vehicle_features.size(2)


        # print(f"customer_features shape: {customer_features.shape}")
        # print(f"vehicle_features shape: {vehicle_features.shape}")
        # customer_features shape: torch.Size([64, 10, 14])
        # vehicle_features shape: torch.Size([64, 1, 2])
        
        # Average customer features
        avg_customer = torch.mean(customer_features, dim=1)
        
        # Average vehicle features
        avg_vehicle = torch.mean(vehicle_features, dim=1)

        # print(f"avg_customer shape: {avg_customer.shape}")
        # print(f"avg_vehicle shape: {avg_vehicle.shape}")
        # avg_customer shape: torch.Size([64, 14])
        # avg_vehicle shape: torch.Size([64, 2])
        
        # Concatenate features
        x = torch.cat([avg_customer, avg_vehicle], dim=1)
        # x shape after concatenation: torch.Size([64, 16])

        # print(f"x shape after concatenation: {x.shape}")

        # print("------------- Start of adaptative input_dim for self.fc1-------------")
        # print(f"self.fc1.weight shape: {self.fc1.weight.shape}")
        # print(f"customer_features size 2: {customer_feautre_dim}")
        # print(f"vehicle_features size 2: {vehicle_feature_dim}")
        
        [embedding_dim, _] = self.fc1.weight.shape
        input_dim = customer_feautre_dim + vehicle_feature_dim
        # print(f"input_dim: {input_dim}, embedding_dim: {embedding_dim}")
        self.fc1 = nn.Linear(input_dim, embedding_dim)
        # print(f"self.fc1.weight shape after adaptation: {self.fc1.weight.shape}")
        # print("------------------------------------------------------------")
        
        # Forward pass through network
        # x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # x = self.fc3(x)  # No activation for final output

        
        # Forward pass through network
        try:
            x = self.relu(self.fc1(x))
        except Exception as e:
            print("Error in BaselineModel forward function: self.fc1(x)")
            print("self.fc1 =", self.fc1)
            raise e
        
        try:
            x = self.relu(self.fc2(x))
        except Exception as e:
            print("Error in BaselineModel forward function: x = self.relu(self.fc2(x))")
            raise e
        
        try:
            x = self.fc3(x)  # No activation for final output
        except Exception as e:
            print("Error in BaselineModel forward function: x = self.relu(self.fc1(x))")
            raise e
        
        return x