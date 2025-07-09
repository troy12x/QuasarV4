import torch
import torch.nn as nn
from transformers import (
    SiglipForImageClassification, 
    SiglipImageProcessor,  # Use image processor directly
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process image using image processor directly
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Remove batch dimension
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        # Handle label mapping: dataset has 0=AI, 1=Real but model expects 0=real, 1=AI
        # So we need to flip the labels
        original_label = item['label']
        mapped_label = 1 - original_label  # Flip: 0->1, 1->0
        
        result = {
            'pixel_values': pixel_values,
            'labels': torch.tensor(mapped_label, dtype=torch.long)
        }
        # print(f"Dataset returning keys: {list(result.keys())}") # Debug print
        return result

def compute_metrics(eval_pred):
    # print("--- Inside compute_metrics ---") # Debug print
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    # Configuration
    model_name = "eyad-silx/SI-IMAGE"
    dataset_name = "Hemg/AI-Generated-vs-Real-Images-Datasets"
    output_dir = "./siglip-lora-finetuned"
    
    # Load model and image processor separately
    logger.info("Loading model and image processor...")
    model = SiglipForImageClassification.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Use SiglipImageProcessor directly to avoid tokenizer issues
    processor = SiglipImageProcessor.from_pretrained(model_name)
    logger.info("Successfully loaded SiglipImageProcessor")
    
    # LoRA configuration
    lora_config = LoraConfig(

        inference_mode=False,
        r=16,  # Low rank adaptation dimension
        lora_alpha=32,  # LoRA scaling parameter
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
    )
    
    # Apply LoRA to model
    logger.info("Applying LoRA configuration...")
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset(dataset_name)
    
    # Use only first 100 samples
    logger.info("Using only first 100 samples from dataset...")
    if 'train' in dataset:
        small_dataset = dataset['train'].select(range(5000))
    else:
        # If dataset structure is different, take first 100 from available split
        available_split = list(dataset.keys())[0]
        small_dataset = dataset[available_split].select(range(5000))
    
    # Create train/validation split from the 100 samples
    # Use 80 samples for training, 20 for validation
    split_dataset = small_dataset.train_test_split(test_size=0.2, seed=42)
    
    # Create datasets
    train_dataset = ImageDataset(split_dataset['train'], processor)  # 80 samples
    val_dataset = ImageDataset(split_dataset['test'], processor)     # 20 samples
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Training arguments to prevent overfitting
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # Reduced epochs to prevent overfitting
        learning_rate=5e-5,  # Lower learning rate for stable training
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        # --- Re-enable evaluation to monitor for overfitting ---
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,

        # --- End of evaluation settings ---
        report_to="tensorboard",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        remove_unused_columns=False,
        label_names=["labels"],  # IMPORTANT: Must match the key from ImageDataset
        push_to_hub=False,
        hub_model_id=None,
    )

    # Initialize trainer with evaluation and early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the model
    logger.info("Saving model...")
    trainer.save_model()

    # Evaluate on validation set
    logger.info("Final evaluation...")
    eval_results = trainer.evaluate()
    logger.info(f"Final evaluation results: {eval_results}")
    

    
    # Save LoRA adapter
    model.save_pretrained(f"{output_dir}/lora_adapter")
    logger.info(f"LoRA adapter saved to {output_dir}/lora_adapter")

if __name__ == "__main__":
    main()