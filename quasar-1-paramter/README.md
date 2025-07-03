# Quasar-1-Parameter LNN

This project implements a Liquid Neural Network (LNN) with only a single learnable parameter. The goal is to explore whether complex learning and intelligence can emerge from rich temporal dynamics modulated by a single scalar, rather than from a vast number of stored weights.

## Architecture

The model, `Quasar1ParameterLNN`, is defined in `model.py`. Its key features are:

- **Single Learnable Parameter (`lambda`):** A single scalar `torch.nn.Parameter` that is updated by the optimizer.
- **Fixed Random Projections:** All other components, including the embedding layer and linear transformations, are initialized with random weights and are **not trainable** (`requires_grad=False`).
- **Lambda-Modulated Dynamics:** The core of the recurrent update is an ODE system where `lambda` scales the influence of the recurrent connection, allowing the model to learn the optimal balance between external input and its internal state.

## How to Run

The `train.py` script provides a simple demonstration of how to train the model. It uses a dummy tokenizer and randomly generated data.

To run the training script:

```bash
python train.py --hidden_size 512 --num_layers 4 --learning_rate 0.001 --num_epochs 5
```

During training, you can observe the `lambda` parameter changing as the model learns.

## Goals & Next Steps

- **Demonstrate Learning:** The primary goal is to show that the model's loss decreases and `lambda` converges to a stable value, proving that learning is occurring.
- **Large-Scale Training:** The next step is to adapt this model to a real-world pre-training pipeline, similar to `lnn_pretrain.py`, to train it on trillions of tokens.
- **Dynamic Parameterization:** Future work could explore making `lambda` itself a dynamic function of the input, further increasing the model's expressive power without adding trainable parameters.
