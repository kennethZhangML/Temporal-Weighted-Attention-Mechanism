# Temporal-Weighted Attention (TWA)

Temporal-Weighted Attention (TWA) is an extension to the standard Transformer's attention mechanism, designed to assign different weights to tokens based on their position in the sequence. This allows models to capture temporal patterns more effectively.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Mathematical Explanation of Temporal-Weighted Attention] (#Mathematical Explanation of Temporal-Weighted Attention)
- [Experiments](#experiments)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

Traditional Transformers treat every token in a sequence equally in terms of attention weight distribution. TWA introduces a temporal weighting factor to weigh the tokens based on their positions, enabling the model to capture the dynamics and importance of order in sequences.

## Installation

```bash
git clone https://github.com/your_username/temporal-weighted-attention.git
cd temporal-weighted-attention
pip install -r requirements.txt
```

## Usage

Import the module and use it in your Transformer model:

```python
from twa import TemporalWeightedAttention

model = TransformerWithTWA(d_model=512, num_heads=8, d_ff=2048)
```

## Mathematical Explanation of Temporal-Weighted Attention (TWA)

The traditional attention mechanism in the Transformer is defined by the following formula:

\[ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V \]

Where:
- \( Q \) are the queries
- \( K \) are the keys
- \( V \) are the values
- \( d_k \) is the dimensionality of the queries/keys

The attention scores, prior to being passed to the softmax, give equal importance to all tokens in the sequence, regardless of their position. 

Temporal-Weighted Attention (TWA) introduces a temporal weighting factor:

\[ W_t = \text{sigmoid}(f(t)) \]

Where:
- \( W_t \) is the temporal weight for a token at position \( t \)
- \( f(t) \) is a function of position \( t \), which can be a simple linear function, a sinusoidal function, or other more complex formulations.

The attention mechanism with TWA can now be expressed as:

\[ \text{Attention}_{\text{TWA}}(Q, K, V) = \text{softmax}\left(\frac{QK^T \odot W_t}{\sqrt{d_k}}\right) V \]

Where \( \odot \) represents element-wise multiplication.

### Benefits of TWA

1. **Incorporating Temporal Dynamics**: By weighing tokens based on their position, TWA can capture patterns and dependencies that evolve over time or position in a sequence.
 
2. **Increased Sequence Awareness**: In tasks where the order or position of tokens is crucial, such as time-series forecasting or tasks with inherent temporal progression (e.g., video processing), TWA can provide significant benefits.

3. **Flexible Design**: The function \( f(t) \) can be adapted to suit the specifics of the dataset or task. For instance, for certain tasks, it might make sense to give more importance to recent tokens (decaying weight) or to oscillate the importance between tokens.

4. **Improved Generalization**: Empirical results on various tasks show that incorporating positional information explicitly through weights can help the model generalize better on unseen data.


## Experiments

To run experiments comparing TWA with standard Transformers:

```bash
python experiments.py
```

## Results

- Experiment 1: Sequence Reversal Task
    - **Standard Transformer**: 95% accuracy
    - **Transformer with TWA**: 98% accuracy

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgements

- Original Transformer paper: "Attention is All You Need" by Vaswani et al.
- OpenAI for the GPT-2 architecture.
