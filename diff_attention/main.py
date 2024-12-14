import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from dataset import ArabicQADataset
from models import create_qa_model
from diff_train import TrainerDiff
import logging
from pathlib import Path
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_dataloaders(tokenizer, batch_size=8):
    """Set up training and validation dataloaders."""
    try:
        # Make sure tokenizer has padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create datasets
        logger.info("Loading training dataset...")
        train_dataset = ArabicQADataset(
            data_path='data/train-open.json',
            tokenizer=tokenizer,
            max_length=512,
            is_training=True
        )

        logger.info("Loading validation dataset...")
        val_dataset = ArabicQADataset(
            data_path='data/test-open.json',
            tokenizer=tokenizer,
            max_length=512,
            is_training=True
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=train_dataset.collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=val_dataset.collate_fn
        )

        return train_loader, val_loader

    except Exception as e:
        logger.error(f"Error in setup_dataloaders: {str(e)}")
        raise


def train_model(differential_ratio=0.3):
    """Train model with differential attention."""
    config = {
        'model_name': "aubmindlab/aragpt2-base",
        'batch_size': 8,
        'num_epochs': 3,
        'learning_rate': 5e-5,
        'save_path': f"./models/diff_attn_{int(differential_ratio * 100)}",
        'device': "cuda" if torch.cuda.is_available() else "cpu"
    }

    try:
        # Create directories
        Path("./models").mkdir(parents=True, exist_ok=True)

        # Initialize tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

        # Create model with differential attention
        logger.info(f"Creating model with {differential_ratio * 100}% differential attention...")
        model = create_qa_model(
            pretrained_model_name=config['model_name'],
            differential_ratio=differential_ratio
        )

        # Setup dataloaders
        train_loader, val_loader = setup_dataloaders(tokenizer, config['batch_size'])

        # Initialize trainer
        trainer = TrainerDiff(
            model=model,  # Add model here
            model_name=f"diff_attn_{int(differential_ratio * 100)}",
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            save_path=config['save_path'],
            num_epochs=config['num_epochs'],
            learning_rate=config['learning_rate'],
            device=config['device']
        )

        # Train model
        trainer.train()

        # Get final validation loss
        val_loss = trainer.validate()

        return val_loss

    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        raise


def main():
    # Try different ratios of differential attention
    ratios = [0.25, 0.4, 0.5]
    results = {}

    try:
        for ratio in ratios:
            logger.info(f"\nTraining model with {ratio * 100}% differential attention...")
            val_loss = train_model(ratio)
            results[f"differential_{int(ratio * 100)}"] = val_loss

        # Save results
        with open("./models/training_results.json", 'w') as f:
            json.dump(results, f, indent=4)

        # Print results
        logger.info("\nTraining completed! Final results:")
        for model_name, loss in results.items():
            logger.info(f"{model_name}: Validation Loss = {loss:.4f}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()