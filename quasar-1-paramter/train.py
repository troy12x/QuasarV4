import torch
import argparse
from model import Quasar1ParameterLNN
from tqdm import tqdm

# A simple dummy tokenizer for demonstration purposes
class DummyTokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size

    def encode(self, text):
        return torch.randint(0, self.vocab_size, (len(text.split()),))

    def __len__(self):
        return self.vocab_size

def main(args):
    print("--- Initializing 1-Parameter Quasar LNN Training ---")

    # Dummy data and tokenizer
    tokenizer = DummyTokenizer(vocab_size=args.vocab_size)
    dummy_text = " ".join(["token"] * args.sequence_length * 100) # 100 batches of data
    data = tokenizer.encode(dummy_text).view(-1, args.sequence_length)

    # Model Initialization
    model = Quasar1ParameterLNN(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dt=args.dt
    )

    print(f"Model initialized. Total trainable parameters: {model.count_parameters()}")
    assert model.count_parameters() == 1, "Model should only have 1 trainable parameter!"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer and Loss
    optimizer = torch.optim.AdamW([model.lambda_param], lr=args.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    print(f"Starting training on {device}...")

    for epoch in range(args.num_epochs):
        total_loss = 0
        progress_bar = tqdm(data, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch in progress_bar:
            inputs = batch.unsqueeze(0).to(device)
            # Shift labels for next-token prediction
            labels = torch.roll(inputs, shifts=-1, dims=-1)
            
            optimizer.zero_grad()

            logits, _ = model(inputs)

            # Reshape for loss calculation
            loss = loss_fn(logits.view(-1, tokenizer.vocab_size), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item(), "Lambda": model.lambda_param.item()})

        avg_loss = total_loss / len(data)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}, Final Lambda: {model.lambda_param.item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a 1-Parameter Quasar LNN")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden size of the model.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of LNN layers.")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step for Euler integration.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the lambda parameter.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--vocab_size", type=int, default=1000, help="Size of the dummy vocabulary.")
    parser.add_argument("--sequence_length", type=int, default=128, help="Sequence length for training.")
    
    args = parser.parse_args()
    main(args)
