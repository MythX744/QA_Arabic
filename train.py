import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_scheduler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os


class Trainer:
    def __init__(
            self,
            model_name: str,
            train_loader: DataLoader,
            val_loader: DataLoader,
            tokenizer: GPT2Tokenizer,
            save_path: str = "./gpt2_qa_model",
            num_epochs: int = 3,
            learning_rate: float = 5e-5,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.save_path = save_path

        # First freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the last 2 transformer blocks
        for i in range(2):
            for param in self.model.transformer.h[-(i + 1)].parameters():
                param.requires_grad = True

        # Unfreeze the language modeling head (important for generation)
        for param in self.model.lm_head.parameters():
            param.requires_grad = True

        # Initialize optimizer only with unfrozen parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=self.learning_rate)

        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_epochs * len(train_loader)
        )

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.writer = SummaryWriter()

        # Lists to store losses for plotting
        self.train_losses = []
        self.val_losses = []

        # Create plots directory
        os.makedirs("plots", exist_ok=True)

        # Print number of trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%} of total)")

    def train(self):
        """Train the GPT-2 model."""
        print("Starting training...")
        self.model.train()

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_steps = 0

            for batch_idx, batch in enumerate(self.train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                epoch_loss += loss.item()
                epoch_steps += 1

                # Print progress
                if batch_idx % 10 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{self.num_epochs}], "
                        f"Step [{batch_idx}/{len(self.train_loader)}], "
                        f"Loss: {loss.item():.4f}"
                    )

            # Calculate average loss for the epoch
            avg_train_loss = epoch_loss / epoch_steps
            self.train_losses.append(avg_train_loss)

            # Log training loss
            print(f"Epoch [{epoch + 1}/{self.num_epochs}] Average Loss: {avg_train_loss:.4f}")
            self.writer.add_scalar("Loss/train", avg_train_loss, epoch)

            # Validation step
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            print(f"Validation Loss: {val_loss:.4f}")
            self.writer.add_scalar("Loss/val", val_loss, epoch)

            # Save model checkpoint
            self.save_model(epoch)

            # Plot and save losses after each epoch
            self.plot_losses(epoch + 1)

        print("Training complete!")
        self.writer.close()

        # Create final loss plot
        self.plot_losses(self.num_epochs, is_final=True)

    def plot_losses(self, epoch, is_final=False):
        """Plot training and validation losses."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Training Loss', marker='o')
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label='Validation Loss', marker='o')

        plt.title(f'Training and Validation Losses up to Epoch {epoch}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        # Add value labels to points
        for i, (train_loss, val_loss) in enumerate(zip(self.train_losses, self.val_losses)):
            plt.annotate(f'{train_loss:.4f}', (i + 1, train_loss), textcoords="offset points", xytext=(0, 10),
                         ha='center')
            plt.annotate(f'{val_loss:.4f}', (i + 1, val_loss), textcoords="offset points", xytext=(0, -15), ha='center')

        if is_final:
            plt.savefig('plots/final_losses.png')
        else:
            plt.savefig(f'plots/losses_epoch_{epoch}.png')

        plt.close()

    def validate(self):
        """Validate the model on the validation set."""
        self.model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                val_steps += 1

        return val_loss / val_steps

    def save_model(self, epoch):
        """Save the model and tokenizer."""
        save_dir = f"{self.save_path}_epoch_{epoch + 1}"
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"Model saved to {save_dir}")