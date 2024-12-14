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
    # Create dataset for validation
    val_dataset = ArabicQADataset(
        data_path='data/test-open.json',
        tokenizer=tokenizer,
        max_length=128,
        is_training=False  # Set to False for evaluation
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
    Evaluate the model on the validation set using BLEU score.
    """
    model.eval()
    predictions = []
    references = []
    bleu_metric = BLEUScore()

    # Set up generation parameters
    gen_kwargs = {
        'max_length': 64,
        'min_length': 5,
        'num_beams': 4,
        'no_repeat_ngram_size': 3,
        'early_stopping': True,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Generate answers
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )

            # Decode generated answers and get reference answers
            for i, output in enumerate(outputs):
                # Get the original example
                example = val_loader.dataset.get_example_text(batch_idx * val_loader.batch_size + i)

                # Decode the generated answer
                pred_text = tokenizer.decode(output, skip_special_tokens=True)

                # Add to predictions and references
                predictions.append(pred_text)
                references.append(example['answer'].split())  # Split into words for BLEU

                # Print some examples
                if batch_idx == 0 and i < 3:
                    logger.info(f"\nExample {i + 1}:")
                    logger.info(f"Question: {example['question']}")
                    logger.info(f"Generated: {pred_text}")
                    logger.info(f"Reference: {example['answer']}")

    # Calculate BLEU score
    bleu_score = bleu_metric(predictions, references)

    # Save some examples to file
    with open('evaluation_examples.txt', 'w', encoding='utf-8') as f:
        for i in range(min(10, len(predictions))):
            f.write(f"\nExample {i + 1}:\n")
            f.write(f"Generated: {predictions[i]}\n")
            f.write(f"Reference: {' '.join(references[i])}\n")
            f.write("-" * 50 + "\n")

    return bleu_score.item()


def main():
    # Configuration
    config = {
        'model_name': "aubmindlab/aragpt2-base",
        'model_path': "models/gpt2_qa_model_epoch_3",  # Path to your trained model
        'batch_size': 8,
        'device': "cuda" if torch.cuda.is_available() else "cpu"
    }

    logger.info(f"Using device: {config['device']}")

    try:
        # Initialize tokenizer
        logger.info("Initializing tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained(config['model_name'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load the trained model
        logger.info("Loading trained model...")
        model = GPT2LMHeadModel.from_pretrained(config['model_path']).to(config['device'])
        model.eval()

        # Setup validation dataloader
        logger.info("Setting up validation dataloader...")
        val_loader = setup_dataloader(tokenizer, config['batch_size'])

        # Evaluate the model
        logger.info("Starting evaluation...")
        bleu_score = evaluate_model(model, val_loader, tokenizer, config['device'])
        logger.info(f"BLEU Score: {bleu_score:.4f}")

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    main()