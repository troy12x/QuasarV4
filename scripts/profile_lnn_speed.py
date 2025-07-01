import torch
import torch.optim as optim
import sys
import os
from torch.profiler import profile, record_function, ProfilerActivity
from torch.cuda.amp import GradScaler, autocast

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quasar.lnn import LNNModel, LNNConfig

def profile_lnn_training():
    """
    Profiles the LNN model's training step with AMP to identify performance bottlenecks.
    """
    print("--- LNN Training Speed Profiler (AMP Enabled) ---")
    print("Initializing model, optimizer, and dummy data...")

    # 1. Configure the model
    config = LNNConfig(
        vocab_size=50257,
        hidden_size=256,
        num_hidden_layers=2,
        chunk_size=64,
        use_moe=False,
        use_pmb=False
    )
    model = LNNModel(config).cuda()
    model.train()

    # 2. Create dummy data, optimizer, and loss function
    batch_size = 64
    seq_length = 128
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    inputs = {
        "input_ids": torch.randint(0, config.vocab_size, (batch_size, seq_length), device='cuda')
    }
    # Dummy target for loss calculation
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_length), device='cuda')
    loss_fn = torch.nn.CrossEntropyLoss()


    print(f"Model: LNNModel (Training Step)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Input Shape: {inputs['input_ids'].shape}")
    print("-" * 30)

    # 3. Define the training step to profile
    def train_step(model, optimizer, inputs, targets, scaler):
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            outputs = model(**inputs, return_dict=True)
            logits = outputs.logits
            loss = loss_fn(logits.view(-1, config.vocab_size), targets.view(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # 4. Warmup runs
    print("Running warmup steps...")
    for _ in range(5):
        train_step(model, optimizer, inputs, targets, scaler)
    torch.cuda.synchronize()

    # 5. Run the profiler
    print("Starting profiler...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/lnn_amp_profile'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for i in range((1 + 1 + 3) * 2):
            train_step(model, optimizer, inputs, targets, scaler)
            prof.step()

    torch.cuda.synchronize()
    print("Profiling complete.")
    print("-" * 30)

    # 6. Print the results
    print("--- Profiler Results (Top 15 by Self CUDA Time) ---")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))

if __name__ == "__main__":
    profile_lnn_training()
