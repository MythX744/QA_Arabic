import logging
import torch
from torch.utils.data import DataLoader
from torchmetrics.text import BLEUScore
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from diff_dataset import ArabicQADataset
from models import MultiHead
import json
from pathlib import Path
from datetime import datetime
from safetensors.torch import load_file

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_dataloader(tokenizer, batch_size=8):
    """
    Set up test dataloader.
    """
    test_dataset = ArabicQADataset(
        data_path='../data/test-open.json',
        tokenizer=tokenizer,
        max_length=128,
        is_training=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )

    return test_loader


def load_model_with_attention(base_model_name, checkpoint_path, ratio, device):
    """
    Load the complete model with custom attention layers
    """
    try:
        # Load state dict from safetensors
        state_dict = load_file(checkpoint_path / 'model.safetensors', device=device)
        logger.info(f"Loaded state dict with {len(state_dict)} keys")

        # Initialize base model
        model = GPT2LMHeadModel.from_pretrained(base_model_name)

        # Load the complete state dict
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)

        # Replace attention layers
        transformer_blocks = model.transformer.h
        num_layers = len(transformer_blocks)
        layers_to_replace = int(num_layers * ratio)

        for i in range(layers_to_replace):
            logger.info(f"Replacing attention layer {i} with MultiHead")
            transformer_blocks[i].attn = MultiHead(
                dim_model=model.config.n_embd,
                num_heads=model.config.n_head
            ).to(device)

        logger.info(f"Successfully loaded model and replaced {layers_to_replace} attention layers")
        return model

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def evaluate_model(model, test_loader, tokenizer, device, ratio):
    """
    Evaluate the model and save results to JSON.
    """
    model.eval()
    all_results = []
    predictions = []
    references = []
    bleu_metric = BLEUScore()

    gen_kwargs = {
        'max_new_tokens': 64,
        'num_beams': 4,
        'no_repeat_ngram_size': 2,
        'repetition_penalty': 1.2,
        'length_penalty': 1.0,
        'early_stopping': True,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'do_sample': False,
        'use_cache': False  # Temporarily disable caching
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            try:
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # Generate
                try:
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **gen_kwargs
                    )

                    # Check for None
                    if outputs is None or not hasattr(outputs, 'sequences'):
                        logger.error(f"Empty output in batch {batch_idx}")
                        continue

                    generated_sequences = outputs.sequences

                except Exception as gen_error:
                    logger.error(f"Generation error in batch {batch_idx}: {str(gen_error)}")
                    continue

                # Process each example
                for i, output in enumerate(generated_sequences):
                    example_idx = batch_idx * test_loader.batch_size + i
                    example = test_loader.dataset.data[example_idx]
                    question = example['question']
                    reference = example['answer']

                    # Get only the generated part (exclude input)
                    generated_part = output[input_ids.shape[1]:]
                    generated_text = tokenizer.decode(generated_part, skip_special_tokens=True)

                    # Store results
                    result = {
                        "id": example_idx,
                        "question": question,
                        "reference_answer": reference,
                        "generated_answer": generated_text
                    }

                    # Log examples
                    if batch_idx < 2 or batch_idx % 100 == 0:
                        logger.info(f"\nExample {example_idx}:")
                        logger.info(f"Question: {question}")
                        logger.info(f"Generated: {generated_text}")
                        logger.info(f"Reference: {reference}")

                    all_results.append(result)
                    predictions.append(generated_text)
                    references.append([reference])

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                continue

    # Calculate BLEU score
    if predictions:
        try:
            bleu_score = bleu_metric(predictions, references)
        except Exception as e:
            logger.error(f"Error calculating BLEU score: {e}")
            bleu_score = torch.tensor(0.0)
    else:
        logger.warning("No predictions generated, setting BLEU score to 0.0")
        bleu_score = torch.tensor(0.0)

    # Save results
    final_results = {
        "metadata": {
            "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_ratio": ratio,
            "total_examples": len(all_results),
            "bleu_score": float(bleu_score)
        },
        "results": all_results
    }

    results_dir = Path("evaluation_results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = results_dir / f"evaluation_ratio_{ratio}_{timestamp}.json"

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Evaluation results saved to {json_path}")
    return float(bleu_score)


def main():
    config = {
        'model_name': "aubmindlab/aragpt2-base",
        'models_dir': "diff_models",
        'batch_size': 4,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'ratios': [0.25, 0.35, 0.5]
    }

    logger.info(f"Using device: {config['device']}")

    try:
        # Initialize tokenizer
        logger.info("Initializing tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained(config['model_name'])
        tokenizer.padding_side = 'left'  # Set padding side to left
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Setup test dataloader
        test_loader = setup_dataloader(tokenizer, config['batch_size'])

        # Evaluate each model
        results = {}
        for ratio in config['ratios']:
            logger.info(f"\nEvaluating model with ratio {ratio}")

            try:
                # Load model
                model_path = Path(config['models_dir']) / f"ratio_{ratio}_ratio_{ratio}_epoch_3"
                model = load_model_with_attention(config['model_name'], model_path, ratio, config['device'])
                model.eval()

                # Evaluate
                bleu_score = evaluate_model(model, test_loader, tokenizer, config['device'], ratio)
                results[ratio] = bleu_score
                logger.info(f"Ratio {ratio} - BLEU Score: {bleu_score:.4f}")

            except Exception as e:
                logger.error(f"Error evaluating model with ratio {ratio}: {str(e)}")
                results[ratio] = 0.0
                continue

        # Save comparison
        comparison = {
            "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": results
        }

        with open("evaluation_results/comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)

        logger.info("\nEvaluation complete! Summary of results:")
        for ratio, score in results.items():
            logger.info(f"Ratio {ratio}: BLEU = {score:.4f}")

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    main()