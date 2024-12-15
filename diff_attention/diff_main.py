import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from diff_dataset import ArabicQADataset
from diff_train import TrainerDiff
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_dataloaders(tokenizer, batch_size=8):
    """
    Set up training and validation dataloaders.
    """
    # Make sure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create datasets
    train_dataset = ArabicQADataset(
        data_path='../data/train-open.json',
        tokenizer=tokenizer,
        max_length=128,
        is_training=True
    )

    # Use validation set during training
    val_dataset = ArabicQADataset(
        data_path='../data/val-open.json',
        tokenizer=tokenizer,
        max_length=128,
        is_training=False
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


def train_with_ratio(config, train_loader, val_loader, tokenizer, ratio):
    """
    Train model with specific attention layer replacement ratio
    """
    logger.info(f"Training model with {ratio * 100}% attention layers replaced")

    # Create save directory for this ratio
    ratio_save_path = os.path.join(config['save_path'], f"ratio_{ratio}")
    os.makedirs(ratio_save_path, exist_ok=True)

    trainer = TrainerDiff(
        model_name=config['model_name'],  # Use the Arabic GPT-2 model
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        replacement_ratio=ratio,
        save_path=ratio_save_path,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        device=config['device']
    )

    trainer.train()


def main():
    # Training configurations
    config = {
        'model_name': "aubmindlab/aragpt2-base",  # Arabic GPT-2 model
        'batch_size': 8,
        'num_epochs': 3,
        'learning_rate': 5e-5,
        'save_path': "diff_models",  # Changed to match your save path
        'device': "cuda" if torch.cuda.is_available() else "cpu"
    }

    logger.info("Starting training process...")
    logger.info(f"Using device: {config['device']}")

    try:
        # Create main save directory
        os.makedirs(config['save_path'], exist_ok=True)

        # Initialize tokenizer
        logger.info("Initializing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

        # Setup dataloaders
        logger.info("Setting up dataloaders...")
        train_loader, val_loader = setup_dataloaders(tokenizer, config['batch_size'])

        # Train models with different ratios
        ratios = [0.25, 0.35, 0.5]
        for ratio in ratios:
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Starting training for ratio {ratio}")
            logger.info(f"{'=' * 50}")

            try:
                train_with_ratio(config, train_loader, val_loader, tokenizer, ratio)
                logger.info(f"Successfully completed training for ratio {ratio}")
            except Exception as e:
                logger.error(f"Error during training for ratio {ratio}: {str(e)}")
                continue

        logger.info("All training configurations completed!")

    except Exception as e:
        logger.error(f"An error occurred during setup: {str(e)}")
        raise


if __name__ == "__main__":
    main()