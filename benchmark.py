import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from collections import defaultdict
import time

from quasar.lnn import LNNModel, LNNConfig

# Register your model config and class
AutoConfig.register("quasar", LNNConfig)
AutoModelForCausalLM.register(LNNConfig, LNNModel)

# 2. Load the model and tokenizer
model_name = "silx-ai/TARS2"
config     = AutoConfig.from_pretrained(model_name)
model      = AutoModelForCausalLM.from_pretrained(model_name, config=config)
tokenizer  = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# 3. Select math-focused MMLU subjects
math_subjects = [
    "abstract_algebra",
    "elementary_mathematics",
    "high_school_mathematics",
    "college_mathematics",
    "high_school_statistics"
]

# 4. Define loss computation
def compute_choice_loss(question: str, choice: str) -> float:
    text   = f"{question}\nAnswer: {choice}"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return outputs.loss.item()

# 5. Evaluate across selected subjects
total_correct = 0
total_questions = 0
per_subject = defaultdict(lambda: {"correct": 0, "total": 0})

start_time = time.time()
for subject in math_subjects:
    print(f"Evaluating subject: {subject}")
    dataset = load_dataset("edinburgh-dawg/mmlu-redux-2.0", subject, split="test")
    for ex in dataset:
        question   = ex["question"]
        choices    = ex["choices"]
        answer_idx = ex["answer"]
        
        losses = [compute_choice_loss(question, c) for c in choices]
        pred_idx = min(range(len(losses)), key=lambda i: losses[i])

        per_subject[subject]["total"] += 1
        total_questions += 1
        if pred_idx == answer_idx:
            per_subject[subject]["correct"] += 1
            total_correct += 1

elapsed = time.time() - start_time

# 6. Print summary
print(f"\nTotal Math Accuracy: {total_correct}/{total_questions} = {total_correct / total_questions:.4f} (Time: {elapsed:.1f}s)\n")
print("Per-subject accuracy:")
for subj, stats in per_subject.items():
    corr = stats["correct"]
    tot  = stats["total"]
    acc  = corr / tot if tot > 0 else 0.0
    print(f"{subj:25s}: {corr}/{tot} = {acc:.4f}")
