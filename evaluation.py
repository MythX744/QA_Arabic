import logging

import torch
from torch.utils.data import DataLoader
from torchmetrics.text import BLEUScore
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from dataset import ArabicQADataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_dataloader(tokenizer, batch_size=8):
    """
    Set up validation dataloader.
    """
    # Make sure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataset for validation
    val_dataset = ArabicQADataset(
        data_path='data/test-open.json',
        tokenizer=tokenizer,
        max_length=32,
        is_training=False  # No labels for evaluation
    )

    # Create dataloader for validation
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn
    )

    return val_loader


def evaluate_model(model, val_loader, tokenizer, device):
    """
    Evaluate the model on the validation set using BLEU score from torchmetrics.
    """
    model.eval()
    references = []
    predictions = []

    bleu_metric = BLEUScore()

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Generate answers using beam search instead of argmax
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=100,  # Adjust based on your needs
                num_beams=4,  # Beam search for better generation
                no_repeat_ngram_size=3,  # Prevent repetition
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )

            # Decode the generated and true texts
            predicted_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            true_text = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Clean up the predictions (remove the question part)
            cleaned_predictions = [text.split("الجواب:")[-1].strip() for text in predicted_text]

            # Flatten the predictions and references
            predictions.extend(cleaned_predictions)
            references.extend([t.split() for t in true_text])  # Split text into words for BLEU calculation

    # Calculate BLEU score
    bleu_score = bleu_metric(predictions, references)
    print(f"Validation BLEU Score: {bleu_score.item()}")

    # Print some examples
    print("\nSample Predictions:")
    for i in range(min(3, len(predictions))):
        print(f"\nPrediction {i + 1}:")
        print(f"Predicted: {predictions[i]}")
        print(f"Reference: {' '.join(references[i])}")

    return bleu_score


def main():
    # Configuration
    config = {
        'model_name': "aubmindlab/aragpt2-base",
        'model_path': "models/gpt2_qa_model_epoch_3",  # Path to your trained model
        'batch_size': 8,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'max_length': 512  # Increased max length for Arabic text
    }

    logger.info(f"Using device: {config['device']}")

    try:
        # Initialize tokenizer
        logger.info("Initializing tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained(config['model_name'])

        # Ensure tokenizer has padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load the trained model
        logger.info("Loading trained model...")
        model = GPT2LMHeadModel.from_pretrained(config['model_path']).to(config['device'])
        model.eval()

        # Setup validation dataloader
        logger.info("Setting up dataloader for validation...")
        val_dataset = ArabicQADataset(
            data_path='data/test-open.json',
            tokenizer=tokenizer,
            max_length=config['max_length'],
            is_training=True  # Set to True to get labels for BLEU score calculation
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=val_dataset.collate_fn
        )

        # Evaluate the model
        logger.info("Starting evaluation...")
        bleu_score = evaluate_model(model, val_loader, tokenizer, config['device'])

        # Log evaluation results
        logger.info(f"Final BLEU Score: {bleu_score:.4f}")

        # Optional: Generate some example predictions
        logger.info("\nGenerating example predictions...")
        num_examples = 3
        for i in range(min(num_examples, len(val_dataset))):
            example = val_dataset.get_example_text(i)
            input_text = f"السؤال: {example['question']}\nالجواب:"

            # Encode input
            inputs = tokenizer(input_text, return_tensors="pt").to(config['device'])

            # Generate answer
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=config['max_length'],
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    early_stopping=True
                )

            # Decode generated answer
            generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_answer = generated_answer.split("الجواب:")[-1].strip()

            # Print results
            logger.info(f"\nExample {i + 1}:")
            logger.info(f"Question: {example['question']}")
            logger.info(f"Generated Answer: {generated_answer}")
            logger.info(f"True Answer: {example['answer']}")

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
