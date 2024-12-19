import logging
import torch
import os
from torch.utils.data import DataLoader
from torchmetrics.text import BLEUScore
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from diff_dataset import ArabicQADataset
from models import replace_gpt2_attn
import json
from pathlib import Path
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_dataloader(data_path, tokenizer, batch_size=8):
    test_dataset = ArabicQADataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=128,
        is_training=False
    )
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )


def extract_answer(text):
    """Extract answer part from the text."""
    if "Answer:" in text:
        return text.split("Answer:")[-1].strip()
    return text


def evaluate_model(model, val_loader, tokenizer, device):
    model.eval()
    all_results = []

    with torch.no_grad():
        for batch in val_loader:
            # Prepare input
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Get next token predictions
            next_token_logits = outputs.logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            # Process each example in the batch
            for i in range(len(input_ids)):
                current_input = input_ids[i]
                generated_ids = current_input.clone()

                # Generate one token at a time
                for _ in range(50):  # Generate up to 50 new tokens
                    # Get model output for current sequence
                    outputs = model(input_ids=generated_ids.unsqueeze(0))
                    next_token_logits = outputs.logits[0, -1, :]
                    next_token = torch.argmax(next_token_logits)

                    # Add next token to sequence
                    generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)])

                    # Stop if we generate the end token
                    if next_token == tokenizer.eos_token_id:
                        break

                # Decode the sequences
                input_text = tokenizer.decode(current_input, skip_special_tokens=True)
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                reference_text = tokenizer.decode(batch["labels"][i], skip_special_tokens=True)

                # Extract answers
                gen_answer = extract_answer(generated_text)
                ref_answer = extract_answer(reference_text)

                result = {
                    "question": input_text.split("Question:")[-1].split("Answer:")[0].strip(),
                    "generated_answer": gen_answer,
                    "reference_answer": ref_answer
                }
                all_results.append(result)

                # Print first few examples
                if len(all_results) <= 3:
                    print(f"\nExample {len(all_results)}:")
                    print(f"Question: {result['question']}")
                    print(f"Generated: {result['generated_answer']}")
                    print(f"Reference: {result['reference_answer']}")

    # Save results
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    return all_results


def main():
    # Configuration
    model_name = "aubmindlab/aragpt2-base"
    model_path = "saved_models_differential_attention/model_0.25_epoch_3"
    data_path = "../data/test-open.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    base_model = GPT2LMHeadModel.from_pretrained(model_name)
    model = replace_gpt2_attn(base_model, dim_model=768, num_heads=12)

    # Load trained weights
    state_dict = load_file(os.path.join(model_path, "model.safetensors"))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # Setup data loader and evaluate
    val_loader = setup_dataloader(data_path, tokenizer)
    logger.info("Running evaluation...")
    results = evaluate_model(model, val_loader, tokenizer, device)

    print(f"\nEvaluation complete. Processed {len(results)} examples.")
    print("Results saved to evaluation_results.json")


if __name__ == "__main__":
    main()