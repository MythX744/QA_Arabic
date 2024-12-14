import torch
from train import Trainer
from models import create_qa_model  # Import from your models.py
from dataset import ArabicQADataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_and_evaluate_models():
    # Initialize tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/aragpt2-base")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataset = ArabicQADataset(
        data_path='data/train-open.json',
        tokenizer=tokenizer,
        max_length=512,
        is_training=True
    )

    val_dataset = ArabicQADataset(
        data_path='data/test-open.json',
        tokenizer=tokenizer,
        max_length=512,
        is_training=True
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=val_dataset.collate_fn
    )

    # Define different ratios for differential attention
    ratios = [0.25, 0.35, 0.5]  # 25%, 35%, and 50%
    results = {}

    # Train models with different ratios
    for ratio in ratios:
        model_name = f"diff_attn_{int(ratio * 100)}"
        logger.info(f"\nTraining model with {ratio * 100}% differential attention")

        try:
            # Create model with differential attention
            model = create_qa_model(
                pretrained_model_name="aubmindlab/aragpt2-base",
                differential_ratio=ratio
            )

            # Move model to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            # Initialize trainer
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                tokenizer=tokenizer,
                save_path=f"./models/{model_name}",
                num_epochs=3,
                learning_rate=5e-5,
                device=device
            )

            # Train the model
            logger.info(f"Starting training for {model_name}")
            trainer.train()

            # Evaluate the model
            logger.info(f"Evaluating {model_name}")
            val_loss = trainer.validate()

            # Store results
            results[model_name] = {
                "ratio": ratio,
                "validation_loss": val_loss,
                "model_path": f"./models/{model_name}"
            }

            logger.info(f"Completed training for {model_name}")
            logger.info(f"Validation Loss: {val_loss:.4f}")

        except Exception as e:
            logger.error(f"Error training model {model_name}: {str(e)}")
            continue

    # Save results
    results_path = Path("results/differential_attention_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    logger.info("\nFinal Results:")
    for model_name, result in results.items():
        logger.info(f"{model_name}:")
        logger.info(f"  Ratio: {result['ratio'] * 100}%")
        logger.info(f"  Validation Loss: {result['validation_loss']:.4f}")
        logger.info(f"  Model saved at: {result['model_path']}")

    return results
