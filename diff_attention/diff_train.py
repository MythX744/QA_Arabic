import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
            self,
            model,
            model_name: str,
            train_loader: DataLoader,
            val_loader: DataLoader,
            tokenizer,
            save_path: str = "./models/qa_model",
            num_epochs: int = 3,
            learning_rate: float = 5e-5,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.save_path = save_path
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device

        # Move model to device
        self.model = self.model.to(self.device)

        # Initialize optimizer with weight decay
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )

        # Add warmup steps
        num_training_steps = num_epochs * len(train_loader)
        num_warmup_steps = num_training_steps // 10

        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        self.writer = SummaryWriter(log_dir=f"runs/{model_name}")

    def train(self):
        """Train the model."""
        logger.info("Starting training...")
        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            train_steps = 0

            # Training loop
            train_iterator = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
            for batch_idx, batch in enumerate(train_iterator):
                try:
                    # Move batch to device
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = input_ids.clone()  # Use input_ids as labels for next token prediction

                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs[0]  # Get loss from model outputs

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.optimizer.step()
                    self.lr_scheduler.step()

                    total_loss += loss.item()
                    train_steps += 1

                    # Update progress bar
                    train_iterator.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{(total_loss / train_steps):.4f}'
                    })

                except Exception as e:
                    logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                    continue

            # Calculate average training loss
            avg_train_loss = total_loss / train_steps
            self.writer.add_scalar("Loss/train", avg_train_loss, epoch)

            # Validation
            val_loss = self.validate()
            self.writer.add_scalar("Loss/val", val_loss, epoch)

            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            logger.info(f"Average training loss: {avg_train_loss:.4f}")
            logger.info(f"Validation loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(epoch, is_best=True)
            else:
                self.save_model(epoch)

        self.writer.close()
        logger.info("Training completed!")

    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        val_steps = 0

        with torch.no_grad():
            for batch in self.val_loader:
                try:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = input_ids.clone()  # Use input_ids as labels

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs[0]

                    total_loss += loss.item()
                    val_steps += 1

                except Exception as e:
                    logger.error(f"Error in validation batch: {str(e)}")
                    continue

        return total_loss / val_steps if val_steps > 0 else float('inf')

    def save_model(self, epoch, is_best=False):
        """Save the model checkpoint."""
        save_dir = f"{self.save_path}_epoch_{epoch + 1}"
        if is_best:
            save_dir += "_best"

        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logger.info(f"Model saved to {save_dir}")