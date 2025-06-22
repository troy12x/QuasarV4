# Quasar: A New Language Model Architecture

This repository contains the implementation of Quasar, a novel language model architecture designed to replace transformer-based LLMs.

## Core Concepts

Quasar is built on three main components:

1.  **🧠 Liquid Neural Network (LNN):** An ODE-based RNN variant that processes sequences in continuous time.
2.  **🧮 Parameter Memory Bank (PMB):** A hierarchical, associative memory system for infinite context with constant-time retrieval.
3.  **🧩 Semantic Chunking System:** A content-aware text segmentation algorithm that breaks down documents into meaningful chunks.

The goal of Quasar is to achieve unlimited context length, perfect recall, and high semantic understanding without relying on attention mechanisms.

## Project Structure

```
quasar/
├── quasar/
│   ├── __init__.py
│   ├── lnn.py           # Liquid Neural Network
│   ├── pmb.py           # Parameter Memory Bank
│   ├── chunker.py       # Semantic Chunker
│   ├── model.py         # Quasar model
│   └── utils.py         # Utility functions
├── tests/
│   ├── __init__.py
│   ├── test_lnn.py
│   ├── test_pmb.py
│   ├── test_chunker.py
│   └── test_model.py
├── scripts/
│   ├── train.py
│   └── inference.py
├── data/
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd quasar
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Download NLTK data:
    ```python
    import nltk
    nltk.download('punkt')
    ```

## Usage

### Training

To train the Quasar model, run the training script:

```bash
python scripts/train.py
```

### Inference

To run inference with a trained model:

```bash
python scripts/inference.py --prompt "Your prompt here"
```
