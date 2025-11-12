"""Transformer components module.

This module contains all the core components of the transformer architecture:
- config: Model configuration
- attention: Multi-head attention mechanism
- mlp: Feedforward network with configurable activation functions
- layer: Full transformer layer (attention + feedforward)
- embed_unembed: Token embedding and unembedding
- positional_encoding: Positional encoding for sequence information
- cross_entropy_loss: Loss function for language modeling
"""

from transformer_from_scratch.components.config import TransformerConfig
from transformer_from_scratch.components.attention import MultiHeadAttention
from transformer_from_scratch.components.mlp import MLP
from transformer_from_scratch.components.layer import Layer
from transformer_from_scratch.components.embed_unembed import Embed, Unembed
from transformer_from_scratch.components.positional_encoding import PositionalEncoding
from transformer_from_scratch.components.cross_entropy_loss import cross_entropy_loss

__all__ = [
    "TransformerConfig",
    "MultiHeadAttention",
    "MLP",
    "Layer",
    "Embed",
    "Unembed",
    "PositionalEncoding",
    "cross_entropy_loss",
]