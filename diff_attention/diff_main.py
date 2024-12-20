import os
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from diff_dataset import ArabicQADataset
from diff_train import TrainerDiff
from tqdm.auto import tqdm
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/training.log')
    ]
)
logger = logging.getLogger(__name__)

# Set configurations
data_path = "../data"
model_name = "aubmindlab/aragpt2-base"
replacement_ratio = 0.5
output_dir = f"saved_models_differential_attention_ration_{replacement_ratio}"
batch_size = 16
num_epochs = 3
learning_rate = 5e-5


# Printing the progress bar in the terminal :)
class ProgressTracker:
    def __init__(self, num_epochs, num_batches):
        self.epoch_bar = tqdm(total=num_epochs, desc="Epochs", position=0)
        self.batch_bar = tqdm(total=num_batches, desc="Batches", position=1, leave=True)
        self.running_loss = 0
        self.batch_count = 0

    def update_batch(self, loss):
        self.batch_bar.update(1)
        self.running_loss += loss
        self.batch_count += 1
        avg_loss = self.running_loss / self.batch_count
        self.batch_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

    def update_epoch(self):
        self.epoch_bar.update(1)
        self.batch_bar.reset()
        self.running_loss = 0
        self.batch_count = 0


def main():
    # Log start time and configurations
    start_time = datetime.now()
    logger.info(f"Starting training at {start_time}")
    logger.info(f"Configurations:")
    logger.info(f"Model: {model_name}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Replacement ratio: {replacement_ratio}")

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Initialize tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = ArabicQADataset(
        data_path=f"{data_path}/train-open.json",
        tokenizer=tokenizer,
        device=device
    )
    val_dataset = ArabicQADataset(
        data_path=f"{data_path}/val-open.json",
        tokenizer=tokenizer,
        device=device
    )

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=train_dataset.collate_fn,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=val_dataset.collate_fn
    )

    # Initialize progress tracker
    progress = ProgressTracker(num_epochs, len(train_loader))

    # Initialize trainer with progress tracker
    logger.info("Initializing trainer...")
    trainer = TrainerDiff(
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        replacement_ratio=replacement_ratio,
        save_path=output_dir,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        progress_tracker=progress
    )

    try:
        # Start training
        logger.info("Starting training...")
        trainer.train()

        # Log completion
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Training completed at {end_time}")
        logger.info(f"Total training time: {duration}")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    finally:
        # Clean up progress bars
        progress.epoch_bar.close()
        progress.batch_bar.close()


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    main()