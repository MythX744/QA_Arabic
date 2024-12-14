import logging
import torch
from torch.utils.data import DataLoader
from torchmetrics.text import BLEUScore
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from dataset import ArabicQADataset
import json
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_dataloader(tokenizer, batch_size=8):
    """
    Set up validation dataloader.
    """
    val_dataset = ArabicQADataset(
        data_path='data/test-open.json',
        tokenizer=tokenizer,
        max_length=128,
        is_training=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn
    )

    return val_loader


def evaluate_model(model, val_loader, tokenizer, device):
    """
    Evaluate the model and save results to JSON.
    """
    model.eval()
    all_results = []
    predictions = []  # Store raw text predictions
    references = []  # Store raw text references
    bleu_metric = BLEUScore()

    # Set up generation parameters
    gen_kwargs = {
        'max_length': 129,
        'min_length': 5,
        'num_beams': 5,
        'no_repeat_ngram_size': 2,
        'repetition_penalty': 1.2,
        'length_penalty': 1.0,
        'early_stopping': True,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'temperature': 0.7,
        'top_k': 50,
        'top_p': 0.9,
    }

    def clean_generated_text(text, question):
        """Clean up the generated text."""
        if text.startswith(question):
            text = text[len(question):]

        clean_text = ''
        for char in text:
            if ord(char) > 1000000:  # Corrupted character detection
                break
            clean_text += char

        clean_text = ' '.join(clean_text.split())
        return clean_text

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

            # Process each example in the batch
            for i, output in enumerate(outputs):
                # Get the original example
                example = val_loader.dataset.get_example_text(batch_idx * val_loader.batch_size + i)

                # Decode and clean the generated answer
                generated_text = tokenizer.decode(output, skip_special_tokens=True)
                cleaned_text = clean_generated_text(generated_text, example['question'])

                # Store results for this example
                result = {
                    "id": batch_idx * val_loader.batch_size + i,
                    "question": example['question'],
                    "context": example.get('context', ''),
                    "reference_answer": example['answer'],
                    "generated_answer": cleaned_text
                }
                all_results.append(result)

                # Store raw predictions and references for BLEU
                predictions.append(cleaned_text)
                references.append([example['answer']])  # References must be a list of possible references

                # Log some examples
                if batch_idx == 0 and i < 3:
                    logger.info(f"\nExample {i + 1}:")
                    logger.info(f"Question: {example['question']}")
                    logger.info(f"Generated: {cleaned_text}")
                    logger.info(f"Reference: {example['answer']}")

    # Calculate BLEU score
    try:
        bleu_score = bleu_metric(predictions, references)
        logger.info(f"Successfully calculated BLEU score")
    except Exception as e:
        logger.error(f"Error calculating BLEU score: {e}")
        logger.error(f"Sample prediction: {predictions[0] if predictions else 'No predictions'}")
        logger.error(f"Sample reference: {references[0] if references else 'No references'}")
        bleu_score = torch.tensor(0.0)

    # Create results dictionary with metadata
    final_results = {
        "metadata": {
            "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_examples": len(all_results),
            "bleu_score": bleu_score.item() if hasattr(bleu_score, 'item') else float(bleu_score)
        },
        "results": all_results
    }

    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"evaluation_results/evaluation_{timestamp}.json"

    # Create directory if it doesn't exist
    Path("evaluation_results").mkdir(exist_ok=True)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Evaluation results saved to {json_path}")
    return bleu_score.item() if hasattr(bleu_score, 'item') else float(bleu_score)


def main():
    # Configuration
    config = {
        'model_name': "aubmindlab/aragpt2-base",
        'model_path': "models/gpt2_qa_model_epoch_3",
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

        # First: Run the full evaluation on test-open.json
        logger.info("Running full evaluation on test set...")
        val_loader = setup_dataloader(tokenizer, config['batch_size'])
        bleu_score = evaluate_model(model, val_loader, tokenizer, config['device'])
        logger.info(f"BLEU Score: {bleu_score:.4f}")

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
