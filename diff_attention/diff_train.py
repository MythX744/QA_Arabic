import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_scheduler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
from models import MultiHead
import logging
logger = logging.getLogger(__name__)


def replace_attention_layers(model, ratio, dim_model=768, num_heads=12):
    """
    Replace a portion of GPT2's attention layers with our MultiHead implementation
    """
    # Get all transformer blocks
    transformer_blocks = model.transformer.h

    # Calculate number of layers to replace
    num_layers = len(transformer_blocks)
    layers_to_replace = int(num_layers * ratio)

    # Replace the attention layers
    for i in range(layers_to_replace):
        # Create new MultiHead attention with correct parameters
        new_attention = MultiHead(
            dim_model=model.config.n_embd,  # Pass dimension from model config
            num_heads=model.config.n_head    # Pass number of heads from model config
        )
        transformer_blocks[i].attn = new_attention

    return model


class TrainerDiff:
    def __init__(
            self,
            model_name: str,
            train_loader: DataLoader,
            val_loader: DataLoader,
            tokenizer: GPT2Tokenizer,
            replacement_ratio: float = 0.3,
            save_path: str = "diff_models/gpt2_qa_model",
            num_epochs: int = 3,
            learning_rate: float = 5e-5,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        # Load the base model
        base_model = GPT2LMHeadModel.from_pretrained(model_name)

        # Replace attention layers with our implementation
        self.model = replace_attention_layers(
            base_model,
            ratio=replacement_ratio,
            dim_model=base_model.config.n_embd,
            num_heads=base_model.config.n_head
        ).to(device)

        self.replacement_ratio = replacement_ratio
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.save_path = save_path

        # Initialize optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_epochs * len(train_loader)
        )

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.writer = SummaryWriter(comment=f'_ratio_{replacement_ratio}')

        # Lists to store losses for plotting
        self.train_losses = []
        self.val_losses = []

        # Create plots directory with ratio information
        self.plot_dir = f"plots_ratio_{replacement_ratio}"
        os.makedirs(self.plot_dir, exist_ok=True)

    def train(self):
        """Train the model."""
        logger.info("Starting training...")

        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            # In the train method of TrainerDiff
            for batch_idx, batch in enumerate(self.train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                train_loss += loss.item()

                if batch_idx % 100 == 0:
                    logger.info(
                        f'Epoch: {epoch + 1}/{self.num_epochs} | Batch: {batch_idx}/{len(self.train_loader)} | Loss: {loss.item():.4f}')

            avg_train_loss = train_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)

            # Validation
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            # Logging
            logger.info(f'Epoch: {epoch + 1}/{self.num_epochs}')
            logger.info(f'Average Train Loss: {avg_train_loss:.4f}')
            logger.info(f'Validation Loss: {val_loss:.4f}')

            # Save model
            self.save_model(epoch)

            # Plot losses
            self.plot_losses(epoch)

    def validate(self):
        """Validate the model."""
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                val_loss += outputs.loss.item()

        return val_loss / len(self.val_loader)

    def plot_losses(self, epoch):
        """Simple plot of training and validation losses."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title(f'Training Progress (Ratio {self.replacement_ratio})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.plot_dir}/losses_epoch_{epoch + 1}.png')
        plt.close()

    def save_model(self, epoch):
        """Save the model and tokenizer with ratio information."""
        save_dir = f"{self.save_path}_ratio_{self.replacement_ratio}_epoch_{epoch + 1}"
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        # Save additional information
        torch.save({
            'replacement_ratio': self.replacement_ratio,
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, os.path.join(save_dir, 'training_info.pt'))

        print(f"Model saved to {save_dir}")