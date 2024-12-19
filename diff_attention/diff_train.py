import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_scheduler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
from models import MultiHead, replace_gpt2_attn
import logging
logger = logging.getLogger(__name__)


class TrainerDiff:
    def __init__(
            self,
            model_name: str,
            train_loader: DataLoader,
            val_loader: DataLoader,
            tokenizer: GPT2Tokenizer,
            replacement_ratio: float = 0.5,
            save_path: str = "diff_models/gpt2_qa_model",
            num_epochs: int = 3,
            learning_rate: float = 5e-5,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            progress_tracker=None
    ):
        self.replacement_ratio = replacement_ratio
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.save_path = save_path
        self.progress_tracker = progress_tracker

        # Load the base model
        logger.info("Loading base model...")
        base_model = GPT2LMHeadModel.from_pretrained(model_name)

        # Replace attention layers first
        logger.info("Replacing attention layers...")
        self.model = replace_gpt2_attn(base_model, dim_model=768, num_heads=12, replacement_ratio=self.replacement_ratio)
        self.model = self.model.to(device)

        # Freeze the non-attention layers
        logger.info("Setting up parameter freezing...")
        for name, param in self.model.named_parameters():
            # Freeze all parameters except the new attention layers
            if 'attn' not in name:  # This will catch both MultiHead and MultiheadDiffAttn parameters
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Verify trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%})")

        # Print which layers are trainable for verification
        logger.info("Trainable layers:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                logger.info(f"  {name}")

        # Initialize optimizer with only trainable parameters
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate
        )

        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_epochs * len(train_loader)
        )

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.writer = SummaryWriter(comment=f'_ratio_{replacement_ratio}')

        self.train_losses = []
        self.val_losses = []
        self.plot_dir = f"plots_ratio_{replacement_ratio}"
        os.makedirs(self.plot_dir, exist_ok=True)

    def train(self):
        """Train the model."""
        logger.info("Starting training...")

        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_loss = 0

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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.lr_scheduler.step()

                train_loss += loss.item()

                # Simple progress print
                print(f"\rEpoch {epoch + 1}: {batch_idx + 1}/{len(self.train_loader)}", end="")

            print()  # New line after each epoch
            avg_train_loss = train_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)

            # Validation
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            # Logging
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
        """Save the model and training information."""
        # Create models directory if it doesn't exist
        os.makedirs('saved_models_differential_attention', exist_ok=True)

        # Simple model name format
        model_name = f'saved_models_differential_attention/model_{self.replacement_ratio}_epoch_{epoch + 1}'

        # Save model and tokenizer
        self.model.save_pretrained(model_name)
        self.tokenizer.save_pretrained(model_name)

        # Save training info
        training_info = {
            'epoch': epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(training_info, f'{model_name}/training_info.pt')

        logger.info(f"Model saved: {model_name}")
