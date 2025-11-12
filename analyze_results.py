"""Analyze and visualize results from experiments."""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_results():
    """Load all result files."""
    results_dir = Path('results')
    all_results = {}
    
    for result_file in results_dir.glob('results_*.json'):
        # Extract activation from filename
        activation = result_file.stem.split('_')[1]
        
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        all_results[activation] = data
    
    return all_results

def create_comparison_table(results):
    """Create final comparison table."""
    comparison_data = []
    
    for activation, data in results.items():
        final_epoch = data[-1]
        comparison_data.append({
            'Activation': activation.upper(),
            'Final Train Loss': f"{final_epoch['train_loss']:.4f}",
            'Final Val Loss': f"{final_epoch['val_loss']:.4f}",
            'Val Perplexity': f"{final_epoch['val_perplexity']:.2f}",
            'Bits/Byte': f"{final_epoch['val_bits_per_byte']:.4f}"
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Val Perplexity')
    
    return df

def plot_results(results):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training loss
    for activation, data in results.items():
        epochs = [d['epoch'] for d in data]
        train_loss = [d['train_loss'] for d in data]
        axes[0, 0].plot(epochs, train_loss, marker='o', label=activation.upper())
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Validation loss
    for activation, data in results.items():
        epochs = [d['epoch'] for d in data]
        val_loss = [d['val_loss'] for d in data]
        axes[0, 1].plot(epochs, val_loss, marker='o', label=activation.upper())
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Validation Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Perplexity
    for activation, data in results.items():
        epochs = [d['epoch'] for d in data]
        perplexity = [d['val_perplexity'] for d in data]
        axes[1, 0].plot(epochs, perplexity, marker='o', label=activation.upper())
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Perplexity')
    axes[1, 0].set_title('Validation Perplexity')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Bits per byte
    for activation, data in results.items():
        epochs = [d['epoch'] for d in data]
        bpb = [d['val_bits_per_byte'] for d in data]
        axes[1, 1].plot(epochs, bpb, marker='o', label=activation.upper())
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Bits per Byte')
    axes[1, 1].set_title('Bits per Byte')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved training_curves.png")
    plt.close()

def main():
    print("Loading results...")
    results = load_results()
    print(f"Found {len(results)} experiments: {list(results.keys())}")
    
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    df = create_comparison_table(results)
    print(df.to_string(index=False))
    print("="*60)
    
    # Save table
    df.to_csv('results/final_comparison.csv', index=False)
    print("\n✓ Saved final_comparison.csv")
    
    # Create plots
    print("\nGenerating plots...")
    plot_results(results)
    
    print("\n✓ Analysis complete!")
    print("Check 'results/' folder for:")
    print("  - final_comparison.csv")
    print("  - training_curves.png")

if __name__ == '__main__':
    main()