#!/bin/bash

echo "Running activation function experiments..."

# Test each activation with 3 epochs on WikiText-2
for activation in relu gelu silu; do
    echo ""
    echo "=========================================="
    echo "Training with $activation"
    echo "=========================================="
    python train_simple.py \
        --activation $activation \
        --num_epochs 3 \
        --batch_size 8
    
    echo "Completed $activation"
done

echo ""
echo "All experiments complete!"