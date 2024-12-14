import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_scheduler
from torch.utils.tensorboard import SummaryWriter


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
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_epochs * len(train_loader)
        )
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.writer = SummaryWriter()

    def train(self):
        """
        Train the GPT-2 model.
        """
        print("Starting training...")
        self.model.train()

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0

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

                # Add gradient clipping here
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.lr_scheduler.step()

                epoch_loss += loss.item()

                # Print progress
                if batch_idx % 10 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{self.num_epochs}], Step [{batch_idx}/{len(self.train_loader)}], Loss: {loss.item():.4f}")

            # Log the epoch loss
            avg_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}] Average Loss: {avg_loss:.4f}")
            self.writer.add_scalar("Loss/train", avg_loss, epoch)

            # Validation step
            val_loss = self.validate()
            print(f"Validation Loss: {val_loss:.4f}")
            self.writer.add_scalar("Loss/val", val_loss, epoch)

            # Save model checkpoint
            self.save_model(epoch)

        print("Training complete!")
        self.writer.close()

    def validate(self):
        """
        Validate the model on the validation set.

        Returns:
            float: Validation loss.
        """
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

        return val_loss / len(self.val_loader)

    def save_model(self, epoch):
        """
        Save the model and tokenizer after each epoch.

        Args:
            epoch (int): Current epoch number.
        """
        save_dir = f"{self.save_path}_epoch_{epoch + 1}"
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"Model saved to {save_dir}")
