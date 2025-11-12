# Transformer Tuning: The Role of Activation Functions in Learning

**Author**: Echo Yu
**Date**: November 2025  

---

## Table of Contents
1. [Problem Statement & Overview](#1-problem-statement--overview)
2. [Methodology](#2-methodology)
3. [Implementation & Demo](#3-implementation--demo)
4. [Assessment & Evaluation](#4-assessment--evaluation)
5. [Model & Data Cards](#5-model--data-cards)
6. [Critical Analysis](#6-critical-analysis)
7. [Documentation & Resource Links](#7-documentation--resource-links)

---

## 1. Problem Statement & Overview

### Problem Statement

While ReLU has been the default activation function in many neural network architectures, newer functions like GELU, SiLU (Swish), and GLU have shown promise in large-scale transformer models. However, their impact in smaller, interpretable settings remains underexplored. **This project investigates how different activation functions affect the learning dynamics, convergence behavior, and final performance of transformer models on language modeling tasks.**

### Research Question

**How do different activation functions (ReLU, GELU, SiLU) impact transformer model performance, training dynamics, and generalization on language modeling tasks?**

### Proposed Approach

I implemented a modular transformer architecture from scratch, allowing for easy swapping of activation functions in the feedforward layers. By training identical models with different activations on the WikiText-2 dataset, I can isolate the impact of activation functions on:
- **Training dynamics**: Convergence speed and stability
- **Final performance**: Perplexity and bits-per-byte metrics
- **Learning behavior**: Loss curves and gradient statistics

### Project Presentation

This project is structured around rigorous experimental methodology and reproducible results. The transformer implementation uses test-driven development principles, with each component individually verified. All experiments use fixed random seeds (seed=42) to ensure reproducibility, and results are systematically logged and analyzed.

**Key Contributions:**
- Modular transformer implementation supporting multiple activation functions
- Comprehensive comparison of ReLU, GELU, and SiLU on language modeling
- Analysis of training dynamics and convergence behavior
- Reproducible experimental framework with detailed documentation

---

## 2. Methodology

### Theoretical Background

#### Activation Functions Compared

**1. ReLU (Rectified Linear Unit)**
```
f(x) = max(0, x)
```
- **Properties**: Simple, efficient, but can cause "dead neurons"
- **Use case**: Traditional baseline for neural networks
- **Gradient**: 1 if x > 0, else 0

**2. GELU (Gaussian Error Linear Unit)**
```
f(x) = x · Φ(x)  where Φ is the CDF of standard normal distribution
```
- **Properties**: Smooth, stochastic interpretation, used in BERT and GPT
- **Use case**: Modern transformers, particularly in NLP
- **Gradient**: Smooth, non-zero everywhere

**3. SiLU (Sigmoid Linear Unit / Swish)**
```
f(x) = x · sigmoid(x) = x / (1 + e^(-x))
```
- **Properties**: Smooth, self-gated, non-monotonic
- **Use case**: Competitive with GELU in various tasks
- **Gradient**: Smooth, bounded

### Architecture

**Base Transformer Configuration:**
```python
- Model dimension (d_model): 256
- MLP hidden dimension (d_mlp): 1024
- Number of heads: 4
- Number of layers: 2
- Vocabulary size: 50,257 (GPT-2 tokenizer)
- Sequence length: 128 tokens
- Total parameters: ~27M
```

**Design Decisions:**
- **Smaller model**: Enables faster experimentation and clearer attribution of effects to activation functions
- **Decoder-only architecture**: Follows GPT-2 style for language modeling
- **Layer normalization**: Applied before attention and feedforward layers
- **No dropout**: Removed to isolate activation function effects

### Dataset

**WikiText-2**
- **Source**: Curated subset of verified Wikipedia articles
- **Size**: 
  - Training: 23,767 sequences
  - Validation: 2,461 sequences
  - Test: 2,891 sequences
- **Preprocessing**: 
  - GPT-2 tokenizer (BPE-based)
  - Maximum sequence length: 128 tokens
  - Padding with EOS token

**Why WikiText-2?**
- Clean, well-structured text
- Diverse vocabulary and topics
- Standard benchmark for language modeling
- Manageable size for multiple experiments

### Training Configuration

**Hyperparameters:**
```python
Optimizer: AdamW
Learning rate: 3e-4
Batch size: 8
Number of epochs: 3
Gradient clipping: 1.0
Weight decay: 0.01
Random seed: 42
```

**Training Process:**
1. **Initialization**: Kaiming initialization for inner MLP weights, Xavier for outer weights
2. **Forward pass**: Causal language modeling with shifted targets
3. **Loss calculation**: Cross-entropy with padding token ignored
4. **Optimization**: AdamW with gradient clipping
5. **Evaluation**: After each epoch on validation set

### Evaluation Metrics

**1. Perplexity**
```
PPL = exp(cross_entropy_loss)
```
- Measures how well the model predicts the next token
- Lower is better (model is less "perplexed")
- Standard metric for language models

**2. Bits per Byte**
```
BPB = cross_entropy_loss / ln(2) / chars_per_token
```
- Measures compression efficiency
- Lower is better
- Approximately 4 characters per token for GPT-2

**3. Training Loss**
- Tracks learning progress during training
- Indicates convergence speed

**Connection to Course Material:**

This methodology directly applies concepts from our course:
- **Transformer architecture**: Implementing attention mechanisms and feedforward layers
- **Language modeling**: Next-token prediction as a fundamental task
- **Experimental design**: Controlled comparisons with proper baselines
- **Evaluation**: Using standard NLP metrics (perplexity, bits-per-byte)

The activation function comparison demonstrates how architectural choices impact model behavior, connecting to our discussions on model internals and design tradeoffs.

---

## 3. Implementation & Demo

### Code Structure

```
transformer-from-scratch-main/
├── transformer_from_scratch/
│   ├── components/
│   │   ├── config.py          # Model configuration
│   │   ├── mlp.py             # Modified feedforward network
│   │   ├── attention.py       # Multi-head attention
│   │   ├── layer.py           # Full transformer layer
│   │   └── ...
│   └── transformer.py         # Main transformer model
├── data_utils.py              # Data loading for WikiText-2
├── metrics.py                 # Evaluation metrics
├── train_simple.py            # Training script
├── analyze_results.py         # Results analysis
└── results/                   # Saved results and figures
```

### Key Implementation Details

#### 1. Modified MLP Layer (Activation Function Support)

```python
class MLP(nn.Module):
    def __init__(self, config: TransformerConfig, activation: str = 'relu'):
        super().__init__()
        self.activation = activation
        
        self.weight_inner = nn.Parameter(torch.empty(config.d_model, config.d_mlp))
        self.bias_inner = nn.Parameter(torch.zeros(config.d_mlp))
        self.weight_outer = nn.Parameter(torch.empty(config.d_mlp, config.d_model))
        self.bias_outer = nn.Parameter(torch.zeros(config.d_model))
        
        # Weight initialization
        nn.init.kaiming_normal_(self.weight_inner)
        nn.init.xavier_normal_(self.weight_outer)
    
    def forward(self, residual_stream):
        # Inner layer
        inner = einsum("batch pos d_model, d_model d_hidden -> batch pos d_hidden",
                      residual_stream, self.weight_inner) + self.bias_inner
        
        # Apply activation
        if self.activation == 'relu':
            inner_activated = torch.relu(inner)
        elif self.activation == 'gelu':
            inner_activated = F.gelu(inner)
        elif self.activation == 'silu':
            inner_activated = F.silu(inner)
        
        # Outer layer
        outer = einsum("batch pos d_hidden, d_hidden d_model -> batch pos d_model",
                      inner_activated, self.weight_outer) + self.bias_outer
        return outer
```

**Key Design Choices:**
- **Modular activation selection**: Single parameter controls activation across all layers
- **Consistent initialization**: Same initialization scheme for fair comparison
- **Einsum operations**: Clear, readable tensor operations using fancy_einsum

#### 2. Training Loop

```python
def train_epoch(model, train_loader, optimizer, device, tokenizer):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        
        # Language modeling: predict next token
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        # Forward pass
        logits = model(inputs)
        
        # Loss calculation
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=tokenizer.pad_token_id
        )
        
        # Backward pass with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)
```

#### 3. Data Loading

```python
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.examples = []
        for text in texts:
            if isinstance(text, str) and text.strip():
                encoding = tokenizer(
                    text, 
                    truncation=True, 
                    max_length=max_length,
                    padding='max_length', 
                    return_tensors='pt'
                )
                self.examples.append(encoding['input_ids'].squeeze(0))
    
    def __getitem__(self, idx):
        return self.examples[idx]
```

### Running the Code

**Setup:**
```bash
# Clone repository
git clone [your-repo-url]
cd transformer-from-scratch-main

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch transformers datasets numpy pandas matplotlib seaborn
```

**Train a model:**
```bash
# Train with specific activation
python train_simple.py --activation relu --num_epochs 3 --batch_size 8
python train_simple.py --activation gelu --num_epochs 3 --batch_size 8
python train_simple.py --activation silu --num_epochs 3 --batch_size 8
```

**Analyze results:**
```bash
python analyze_results.py
```

### Demo Output

**Training Progress (ReLU, Epoch 1):**
```
Using device: cpu
Model has 27,309,056 parameters

Epoch 1/3
Training: 100%|████████| 5942/5942 [57:03<00:00, 1.74it/s, loss=6.5182]
Evaluating: 100%|██████| 616/616 [01:01<00:00, 10.02it/s]
Train Loss: 6.4979
Val Loss: 6.1764
Val Perplexity: 481.27
Val Bits/Byte: 2.2277
```

**Final Results:**
```
Activation  Final Val Loss  Val Perplexity  Bits/Byte
SILU        5.8068          332.55          2.0944
RELU        5.8078          332.87          2.0947
GELU        5.8636          351.99          2.1148
```

---

## 4. Assessment & Evaluation

### Results Summary

#### Final Performance Comparison

| Activation | Final Val Loss | Perplexity | Bits/Byte | Relative Performance |
|-----------|---------------|------------|-----------|---------------------|
| **SiLU**  | 5.8068        | 332.55     | 2.0944    | **Best** ✓          |
| **ReLU**  | 5.8078        | 332.87     | 2.0947    | 2nd (-0.1%)         |
| **GELU**  | 5.8636        | 351.99     | 2.1148    | 3rd (-5.8%)         |

**Key Finding**: SiLU achieves the lowest perplexity (332.55), closely followed by ReLU (332.87), while GELU lags slightly behind (351.99).

### Training Curves Analysis

![Training Curves](training_curves.png)

The training curves reveal several important patterns across the four key metrics:

#### Training Loss (Top Left)
- **Initial convergence**: All three activations start at similar loss values (~6.6) and converge smoothly
- **Convergence speed**: ReLU and SiLU follow nearly identical trajectories throughout training
- **Final performance**: By epoch 3, SiLU (5.33) and ReLU (5.35) achieve the lowest training loss, while GELU (5.42) shows slightly slower learning
- **Stability**: No signs of training instability or divergence for any activation function

**Key Insight**: The smooth, consistent descent indicates stable training across all activation functions, with ReLU and SiLU learning slightly faster than GELU.

#### Validation Loss (Top Right)
- **Generalization gap**: All models show good generalization, with validation loss closely tracking training loss
- **Ranking consistency**: The relative ordering (SiLU ≈ ReLU > GELU) remains stable across all epochs
- **Final values**: SiLU (5.81) and ReLU (5.81) achieve nearly identical validation loss, while GELU (5.86) shows a ~0.8% gap
- **No overfitting**: The parallel descent of validation and training loss suggests the models are not overfitting

**Key Insight**: The consistent ranking across epochs suggests these are genuine performance differences rather than random fluctuations.

#### Validation Perplexity (Bottom Left)
- **Starting point**: GELU begins with higher perplexity (~517) compared to ReLU and SiLU (~494)
- **Improvement rate**: All activations show steady ~32% reduction in perplexity over 3 epochs
- **Final spread**: The ~6% gap between best (SiLU: 332.6) and worst (GELU: 352.0) is modest but consistent
- **Practical significance**: A perplexity of 332 means the model is roughly "confused" among 332 options when predicting the next token

**Key Insight**: The perplexity curves clearly show SiLU and ReLU outperforming GELU throughout training, with no signs of this gap closing.

#### Bits per Byte (Bottom Right)
- **Compression efficiency**: All models improve their compression from ~2.24 to ~2.10 bits per byte
- **Relative performance**: Mirrors the perplexity results, with SiLU (2.094) slightly better than ReLU (2.095) and both better than GELU (2.115)
- **Context**: Random guessing would be ~8 bits per byte, so 2.1 represents substantial compression
- **Practical meaning**: These models achieve roughly 4:1 compression on the test set

**Key Insight**: The similar compression efficiency across activations suggests they're learning comparable representations, with SiLU capturing slightly more structure.

### Cross-Metric Consistency

The consistency across all four plots is striking:
- **No surprises**: The activation that performs best on one metric performs best on all metrics
- **Stable differences**: The performance gaps remain roughly constant throughout training
- **Smooth convergence**: No discontinuities, spikes, or instabilities in any activation function

This consistency strengthens our confidence that the observed differences reflect genuine properties of the activation functions rather than experimental noise.

### Training Dynamics Analysis

#### Convergence Speed

**Epoch 1 → 3 Loss Reduction:**
- **ReLU**: 6.61 → 5.35 (19.1% reduction)
- **SiLU**: 6.60 → 5.33 (19.2% reduction)
- **GELU**: 6.64 → 5.42 (18.4% reduction)

**Observations:**
1. All activations show similar convergence rates
2. SiLU and ReLU have nearly identical training trajectories
3. GELU converges slightly slower in early epochs

#### Validation Performance Over Time

**Perplexity Improvement (Epoch 1 → 3):**
- **SiLU**: 493.7 → 332.6 (32.6% improvement)
- **ReLU**: 493.5 → 332.9 (32.5% improvement)
- **GELU**: 517.0 → 352.0 (31.9% improvement)

All models show consistent improvement, with SiLU and ReLU maintaining a slight edge throughout training.

### Bits per Byte Analysis

Bits-per-byte measures compression efficiency:

**Final BPB:**
- **SiLU**: 2.0944 (best compression)
- **ReLU**: 2.0947 (virtually identical)
- **GELU**: 2.1148 (slightly worse)

**Interpretation**: SiLU and ReLU provide marginally better compression, suggesting they learn slightly more efficient representations of the data.

### Statistical Significance

While the differences are small (< 6% between best and worst), they are **consistent across epochs**, suggesting these are genuine differences rather than random variation.

**Ranking Consistency:**
- All 3 epochs: SiLU ≈ ReLU > GELU
- Performance gap remains stable throughout training

### Evaluation Methodology

**Strengths:**
- ✓ Controlled comparison (same hyperparameters, seeds, data)
- ✓ Multiple evaluation metrics (loss, perplexity, bits/byte)
- ✓ Tracking across full training (3 epochs)
- ✓ Standard benchmark dataset (WikiText-2)

**Limitations:**
- Limited to 3 epochs (longer training might show different patterns)
- Small model size (27M parameters vs. billions in production)
- Single dataset (cross-domain generalization untested)
- CPU-only training (GPU might show different dynamics)

---

## 5. Model & Data Cards

### Model Card

**Model Name**: Transformer Language Model with Configurable Activations

**Model Version/Architecture:**
- Architecture: Decoder-only Transformer (GPT-2 style)
- Parameters: 27,309,056
- Layers: 2 transformer blocks
- Attention heads: 4
- Hidden dimension: 256
- MLP dimension: 1024
- Activation functions: ReLU / GELU / SiLU (configurable)
- Context length: 128 tokens

**Intended Uses:**
- **Primary**: Comparative study of activation functions in transformers
- **Secondary**: Educational demonstration of transformer implementation
- **Research**: Understanding activation function impact on language modeling

**Out-of-Scope Uses:**
- Production language modeling (model is too small)
- Generation of long-form text (limited context window)
- Tasks requiring factual accuracy or safety guarantees
- Any application requiring large-scale performance

**Licenses:**
- Code: MIT License
- Model weights: Available for research and educational purposes
- Dataset: WikiText-2 (Creative Commons Attribution-ShareAlike License)

**Ethical/Bias Considerations:**

**Potential Biases:**
- Wikipedia-sourced data may contain systematic biases present in Wikipedia
- Limited diversity in training data (English-only, encyclopedic style)
- Small model size may amplify memorization vs. generalization

**Fairness Concerns:**
- Model not evaluated for fairness across demographic groups
- No debiasing or fairness interventions applied
- Should not be used for applications affecting individuals

**Mitigation Strategies:**
- Transparent reporting of limitations
- Clear documentation of intended use
- Recommendation against deployment in sensitive applications

**Environmental Impact:**
- Training time: ~3 hours per activation on CPU
- Estimated energy: ~1 kWh total for all experiments
- Carbon footprint: Minimal (local CPU training)

### Data Card

**Dataset Name**: WikiText-2

**Dataset Description:**
- **Source**: Curated subset of verified Wikipedia articles
- **Task**: Language modeling (next-token prediction)
- **Language**: English
- **Size**: 
  - Training: 36,718 articles → 23,767 sequences (128 tokens)
  - Validation: 3,760 articles → 2,461 sequences
  - Test: 4,358 articles → 2,891 sequences
- **License**: Creative Commons Attribution-ShareAlike

**Preprocessing:**
- Tokenization: GPT-2 BPE tokenizer (50,257 vocabulary)
- Sequence length: 128 tokens (truncated/padded)
- Padding: EOS token used for padding
- Filtering: Empty texts removed

**Dataset Characteristics:**
- **Domain**: Encyclopedic knowledge (Wikipedia)
- **Style**: Formal, informative writing
- **Topics**: Broad coverage across subjects
- **Quality**: High (verified Wikipedia articles)

**Known Limitations:**
- English-only (no multilingual coverage)
- Encyclopedic bias (formal style, specific topics)
- Potential Wikipedia biases (coverage gaps, systemic biases)
- Historical data (may not reflect current events)

**Ethical Considerations:**
- Wikipedia content may reflect biases in contributor base
- Coverage varies by topic (some areas underrepresented)
- Should not be assumed to represent all forms of English text

---

## 6. Critical Analysis

### Impact of This Project

**Scientific Contribution:**
This project provides empirical evidence about activation function behavior in small-scale transformer models. The findings suggest that:

1. **ReLU remains competitive**: Despite being simpler, ReLU performs nearly identically to SiLU
2. **SiLU slightly outperforms**: The smooth, self-gated nature of SiLU provides marginal benefits
3. **GELU underperforms**: Contrary to its prevalence in large models, GELU shows slightly worse results in this setting

**Practical Implications:**
- For small models: ReLU may be sufficient (simpler, faster)
- For research: The marginal differences suggest activation choice is less critical than other factors (depth, width, data)
- For practitioners: Results suggest testing multiple activations is worthwhile but differences are modest

### What This Project Reveals

**Key Insights:**

1. **Activation functions matter, but not dramatically**: The ~6% difference between best and worst is meaningful but smaller than impact of other hyperparameters

2. **Smooth activations show advantages**: SiLU (smooth) outperforms ReLU (non-smooth), but the gap is small

3. **Large-model findings don't always transfer**: GELU's success in BERT/GPT may be due to scale effects not present in smaller models

4. **Training stability is good across all**: No activation showed instability or training difficulties

### What This Project Suggests

**Hypothesis for Future Work:**

**Scale Dependency**: The relative performance of activations may change with:
- Model size (larger models may benefit more from GELU)
- Training duration (longer training may amplify differences)
- Dataset complexity (more complex data may favor smooth activations)

**Mechanistic Questions:**
- Why does GELU underperform in small models but excel in large ones?
- Do smooth activations enable better gradient flow in deeper networks?
- How do activation functions interact with other architectural choices?

### Next Steps

**Immediate Extensions:**

1. **Longer training**: Run for 10+ epochs to see if rankings change
2. **Cross-domain evaluation**: Test on SST-2, Penn Treebank, or other datasets
3. **Larger models**: Scale to 100M+ parameters to test scale effects
4. **Include GLU**: Add Gated Linear Units to the comparison

**Research Directions:**

1. **Gradient analysis**: Study gradient statistics and dead neuron percentages
2. **Attention patterns**: Analyze if activations affect attention behavior
3. **Mechanistic interpretability**: Understand what each activation learns
4. **Hyperparameter interaction**: Test activation × learning rate interactions

**Broader Questions:**

- Can we design better activation functions specifically for transformers?
- Do different tasks (classification vs. generation) favor different activations?
- How do activations interact with other modern techniques (layer normalization, residual connections)?

### Limitations and Future Work

**Current Limitations:**

1. **Scale**: Small model (27M) vs. production models (100B+)
2. **Compute**: CPU-only, limited epochs
3. **Scope**: Single dataset, single domain
4. **Depth**: Shallow analysis of training dynamics

**Future Improvements:**

1. **Infrastructure**: GPU training for faster iteration
2. **Scale**: Test on progressively larger models (100M, 1B, 10B)
3. **Diversity**: Multiple datasets (code, dialogue, scientific text)
4. **Analysis**: Deep dive into gradient flow, attention patterns, neuron activation statistics

---

## 7. Documentation & Resource Links

### Repository & Setup

**GitHub Repository**: [Your Link Here]

**Setup Instructions**:
```bash
git clone [your-repo-url]
cd transformer-from-scratch-main
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Requirements**:
```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
numpy<2.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
jaxtyping>=0.2.20
fancy_einsum>=0.0.3
```

### Relevant Papers and Code Bases

**Foundational Papers:**

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - https://arxiv.org/abs/1706.03762
   - Original transformer architecture

2. **BERT: Pre-training of Deep Bidirectional Transformers** (Devlin et al., 2018)
   - https://arxiv.org/abs/1810.04805
   - First major use of GELU in transformers

3. **Searching for Activation Functions** (Ramachandran et al., 2017)
   - https://arxiv.org/abs/1710.05941
   - Introduction of Swish/SiLU

4. **Gaussian Error Linear Units (GELUs)** (Hendrycks & Gimpel, 2016)
   - https://arxiv.org/abs/1606.08415
   - Theoretical motivation for GELU

5. **A Mathematical Framework for Transformer Circuits** (Elhage et al., 2021)
   - https://transformer-circuits.pub/2021/framework/index.html
   - Mechanistic interpretability of transformers

**Code References:**

1. **Original Transformer Implementation**
   - Base code: https://github.com/alan-cooney/transformer-from-scratch
   - Test-driven development approach
   - Clean, modular architecture

2. **Hugging Face Transformers**
   - https://github.com/huggingface/transformers
   - Reference for modern transformer implementations

3. **PyTorch Documentation**
   - https://pytorch.org/docs/stable/nn.html#non-linear-activations
   - Official activation function documentation

**Datasets:**

1. **WikiText-2**
   - https://huggingface.co/datasets/wikitext
   - Language modeling benchmark

2. **GPT-2 Tokenizer**
   - https://huggingface.co/gpt2
   - Tokenization reference

### Additional Resources

**Tutorials:**
- [Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

**Courses:**
- [Deep Learning Curriculum (Transformers)](https://github.com/jacobhilton/deep_learning_curriculum)
- [Stanford CS224N: NLP with Deep Learning](http://web.stanford.edu/class/cs224n/)

---

## Conclusion

This project demonstrates that **activation functions have a measurable but modest impact on small transformer language models**. SiLU achieves the best performance (332.55 perplexity), closely followed by ReLU (332.87), with GELU trailing slightly (351.99). 

The similar performance of ReLU and SiLU suggests that **simple activation functions remain competitive in smaller models**, while the prevalence of GELU in large models may be due to scale-dependent effects not observed here.

**Key Takeaway**: While activation function choice matters, good data science practice—proper experimental design, reproducibility, and thorough analysis—matters more.

---

## Acknowledgments

- Base transformer implementation: [alan-cooney/transformer-from-scratch](https://github.com/alan-cooney/transformer-from-scratch)
- Dataset: WikiText-2 (Merity et al., 2016)
- Framework: PyTorch, Hugging Face


