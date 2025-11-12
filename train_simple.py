"""Simple training script for activation comparison."""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import argparse
import json
from pathlib import Path

from transformer_from_scratch.transformer import Transformer
from transformer_from_scratch.components.config import TransformerConfig
from data_utils import get_wikitext2_dataloaders
from metrics import calculate_perplexity, calculate_bits_per_byte

def train_epoch(model, train_loader, optimizer, device, tokenizer):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        batch = batch.to(device)
        
        # Prepare inputs and targets
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        # Forward pass
        logits = model(inputs)
        
        # Calculate loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=tokenizer.pad_token_id
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches

@torch.no_grad()
def evaluate(model, data_loader, device, tokenizer):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(data_loader, desc='Evaluating'):
        batch = batch.to(device)
        
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        logits = model(inputs)
        
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=tokenizer.pad_token_id
        )
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    return {
        'loss': avg_loss,
        'perplexity': calculate_perplexity(avg_loss),
        'bits_per_byte': calculate_bits_per_byte(avg_loss)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation', type=str, default='relu', 
                       choices=['relu', 'gelu', 'silu'])
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Config with smaller model for testing
    config = TransformerConfig(
        d_model=256,
        d_mlp=1024,
        n_heads=4,
        n_layers=2,
        activation=args.activation
    )
    
    # Load data
    train_loader, val_loader, test_loader, tokenizer = get_wikitext2_dataloaders(
        batch_size=args.batch_size,
        max_length=128
    )
    
    # Create model
    model = Transformer(config).to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=3e-4)
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    results = []
    
    # Training loop
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device, tokenizer)
        val_metrics = evaluate(model, val_loader, device, tokenizer)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Perplexity: {val_metrics['perplexity']:.2f}")
        print(f"Val Bits/Byte: {val_metrics['bits_per_byte']:.4f}")
        
        # Save epoch results
        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_perplexity': val_metrics['perplexity'],
            'val_bits_per_byte': val_metrics['bits_per_byte']
        })
    
    # Save all results to file
    results_file = f'results/results_{args.activation}_wikitext2.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"✓ Results saved to {results_file}")

if __name__ == '__main__':
    main()