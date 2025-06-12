import torch

def inspect_tensor_stats(filepath):
    print(f"\nInspecting {filepath}...")
    checkpoint = torch.load(filepath, map_location="cpu")

    for name, tensor in checkpoint.items():
        if isinstance(tensor, torch.Tensor):
            shape = tuple(tensor.shape)
            t_min = tensor.min().item()
            t_max = tensor.max().item()
            print(f"{name}: shape={shape}, min={t_min:.4f}, max={t_max:.4f}")
        else:
            print(f"{name}: Not a tensor, type={type(tensor)}")

# Update paths if needed
inspect_tensor_stats("checkpoints/model_final_baseline.pt")
inspect_tensor_stats("checkpoints/model_final_policy.pt")
