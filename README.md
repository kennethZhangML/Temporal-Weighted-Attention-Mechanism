# Temporal-Weighted Attention (TWA)

Temporal-Weighted Attention (TWA) is an extension to the standard Transformer's attention mechanism, designed to assign different weights to tokens based on their position in the sequence. This allows models to capture temporal patterns more effectively.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
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
