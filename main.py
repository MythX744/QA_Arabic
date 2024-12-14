import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from dataset import ArabicQADataset
from train import Trainer
import logging

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
        data_path='data/train-open.json',
        tokenizer=tokenizer,
        max_length=32,
        is_training=True
    )

    val_dataset = ArabicQADataset(
        data_path='data/test-open.json',
        tokenizer=tokenizer,
        max_length=32,
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


def main():
    # Training configurations
    config = {
        'model_name': "aubmindlab/aragpt2-base",
        'batch_size': 8,
        'num_epochs': 3,
        'learning_rate': 5e-5,
        'save_path': "./models/gpt2_qa_model",
        'device': "cuda" if torch.cuda.is_available() else "cpu"
    }

    logger.info("Starting training process...")
    logger.info(f"Using device: {config['device']}")

    try:
        # Initialize tokenizer
        logger.info("Initializing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

        # Setup dataloaders
        logger.info("Setting up dataloaders...")
        train_loader, val_loader = setup_dataloaders(tokenizer, config['batch_size'])

        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = Trainer(
            model_name=config['model_name'],
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            save_path=config['save_path'],
            num_epochs=config['num_epochs'],
            learning_rate=config['learning_rate'],
            device=config['device']
        )

        # Start training
        logger.info("Starting training...")
        trainer.train()

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()