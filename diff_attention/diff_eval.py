import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from models import replace_gpt2_attn
from diff_dataset import ArabicQADataset
from torchmetrics.text.bleu import BLEUScore
import json
import logging
import sys
import os
from transformers import GPT2LMHeadModel
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/evaluation.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def load_evaluation_model(model_path, base_model_name, replacement_ratio=0.5):
    """Load the model with proper architecture."""
    logger.info(f"Loading base model architecture...")
    base_model = GPT2LMHeadModel.from_pretrained(base_model_name)

    # Apply the architecture transformation
    model = replace_gpt2_attn(base_model, dim_model=768, num_heads=12, replacement_ratio=replacement_ratio)

    # Load the trained weights
    logger.info(f"Loading weights from {model_path}")
    state_dict = torch.load(os.path.join(model_path, 'model.pt'), map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    logger.info("Model loaded successfully")
    return model


def get_next_token_prob(model, input_ids, attention_mask):
    """Get probabilities for next token."""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_probs = F.softmax(next_token_logits, dim=-1)
    return next_token_probs


def generate_answer_token_by_token(model, tokenizer, question, device, max_length=100):
    """Generate answer one token at a time."""
    model.eval()

    # Prepare initial input
    prompt = f"Question: {question}\nAnswer:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Initialize tensors
    current_length = input_ids.shape[1]
    attention_mask = torch.ones_like(input_ids)

    # Generate tokens
    for _ in range(max_length):
        probs = get_next_token_prob(model, input_ids, attention_mask)
        next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)

        # Break if EOS token
        if next_token.item() == tokenizer.eos_token_id:
            break

        # Append new token
        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.cat([attention_mask,
                                    torch.ones((1, 1), device=device)], dim=1)

    # Decode and extract answer
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    if "Answer:" in generated_text:
        answer = generated_text.split("Answer:")[1].strip()
    else:
        answer = generated_text.strip()

    return answer


def evaluate_model(model, tokenizer, test_dataset, device):
    """Main evaluation function"""
    logger.info("Starting evaluation...")
    model.eval()
    bleu_score = BLEUScore()

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one at a time
        collate_fn=test_dataset.collate_fn,
        shuffle=False
    )

    all_predictions = []
    all_references = []
    all_questions = []

    for batch_idx, batch in enumerate(test_loader):
        try:
            # Extract question and reference
            text = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)
            if "Question:" in text and "Answer:" in text:
                question = text.split("Question:")[1].split("Answer:")[0].strip()
                reference = text.split("Answer:")[1].strip()
            else:
                continue

            # Generate answer
            prediction = generate_answer_token_by_token(model, tokenizer, question, device)

            # Store results
            all_questions.append(question)
            all_predictions.append(prediction)
            all_references.append([reference])

            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"\nProcessed example {batch_idx}")
                logger.info(f"Q: {question}")
                logger.info(f"Generated A: {prediction}")
                logger.info(f"Reference A: {reference}")

        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {str(e)}")
            continue

    # Calculate BLEU score
    score = bleu_score(all_predictions, all_references)
    logger.info(f"Final BLEU Score: {score:.4f}")

    # Save results
    results = {
        "bleu_score": float(score),
        "samples": [
            {
                "question": q,
                "prediction": p,
                "reference": r[0]
            }
            for q, p, r in zip(all_questions[:20], all_predictions[:20], all_references[:20])
        ]
    }

    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    return score


if __name__ == "__main__":
    # Configuration
    test_data_path = "../data/test-open.json"
    base_model_name = "aubmindlab/aragpt2-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load test dataset
    logger.info("Loading test dataset...")
    test_dataset = ArabicQADataset(
        data_path=test_data_path,
        tokenizer=tokenizer,
        max_length=512,
        device=device
    )

    # Find the latest model
    model_dir = "saved_models_differential_attention_ratio_0.5"
    epochs = [d for d in os.listdir(model_dir) if d.startswith('epoch_')]
    latest_epoch = max(epochs, key=lambda x: int(x.split('_')[1]))
    model_path = os.path.join(model_dir, latest_epoch)

    # Load and evaluate model
    logger.info(f"Loading model from {model_path}")
    model = load_evaluation_model(model_path, base_model_name)
    model = model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Run evaluation
    final_bleu_score = evaluate_model(model, tokenizer, test_dataset, device)
    logger.info(f"Evaluation complete. Final BLEU Score: {final_bleu_score:.4f}")