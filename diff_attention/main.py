import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from dataset import ArabicQADataset
from .models import create_qa_model
import logging
from pathlib import Path
import json
from train import Trainer

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

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")

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


def train_differential_model(ratio=0.3):
    """Train model with differential attention at specified ratio."""
    # Create directories if they don't exist
    Path("./models").mkdir(parents=True, exist_ok=True)

    config = {
        'model_name': "aubmindlab/aragpt2-base",
        'batch_size': 8,
        'num_epochs': 3,
        'learning_rate': 5e-5,
        'save_path': f"./models/differential_gpt2_{int(ratio * 100)}",
        'device': "cuda" if torch.cuda.is_available() else "cpu"
    }

    try:
        # Initialize tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

        # Create QA model with differential attention
        logger.info(f"Creating model with {ratio * 100}% differential attention...")
        model = create_qa_model(
            pretrained_model_name=config['model_name'],
            differential_ratio=ratio
        )

        # Log device information
        logger.info(f"Using device: {config['device']}")
        if config['device'] == 'cuda':
            logger.info(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Move model to device
        model = model.to(config['device'])

        # Setup dataloaders
        train_loader, val_loader = setup_dataloaders(tokenizer, config['batch_size'])

        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        model_name = f"diff_attn_{int(ratio * 100)}"

        # Initialize trainer
        trainer = Trainer(
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                tokenizer=tokenizer,
                save_path=f"./models/{model_name}",
                num_epochs=3,
                learning_rate=5e-5,
                device=device
        )

        # Train model
        logger.info("Starting training...")
        trainer.train()

        # Validate final model
        val_loss = trainer.validate()
        logger.info(f"Final validation loss: {val_loss:.4f}")

        return val_loss

    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise


def main():
    # Try different ratios of differential attention
    ratios = [0.25, 0.35, 0.5]  # 25%, 35%, and 50%
    results = {}

    try:
        for ratio in ratios:
            logger.info(f"\nTraining model with {ratio * 100}% differential attention...")
            val_loss = train_differential_model(ratio)
            results[f"differential_{int(ratio * 100)}"] = val_loss

        # Save results to file
        results_file = Path("./models/training_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

        # Print final results
        logger.info("\nTraining completed! Final results:")
        for model_name, loss in results.items():
            logger.info(f"{model_name}: Validation Loss = {loss:.4f}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()