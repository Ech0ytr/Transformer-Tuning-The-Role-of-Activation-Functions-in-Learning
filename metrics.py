"""Metrics for evaluating language models."""

import numpy as np
import torch

def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from cross-entropy loss."""
    return np.exp(loss)

def calculate_bits_per_byte(loss: float) -> float:
    """Calculate bits per byte."""
    bits_per_token = loss / np.log(2)
    return bits_per_token / 4.0  # Assuming ~4 chars per token

def calculate_gradient_stats(model: torch.nn.Module) -> dict:
    """Calculate gradient statistics."""
    total_norm = 0.0
    num_params = 0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            num_params += 1
    
    total_norm = total_norm ** 0.5
    
    return {
        'grad_norm': total_norm,
        'grad_norm_avg': total_norm / max(num_params, 1)
    }

if __name__ == '__main__':
    # Test
    print("Testing metrics...")
    loss = 3.5
    print(f"Loss: {loss} → Perplexity: {calculate_perplexity(loss):.2f}")
    print(f"Loss: {loss} → Bits/Byte: {calculate_bits_per_byte(loss):.4f}")
    print("✓ Metrics work!")