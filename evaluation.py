import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from dataset import ArabicQADataset
from torch.utils.data import DataLoader
from typing import List, Dict
import json
import logging
from pathlib import Path
from torchmetrics.text import BLEUScore, ROUGEScore

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QAEvaluator:
    def __init__(
            self,
            model_path: str,
            test_data_path: str,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the evaluator.

        Args:
            model_path: Path to the saved model
            test_data_path: Path to test data
            device: Device to run the model on
        """
        self.device = device
        self.model_path = model_path
        self.test_data_path = test_data_path

        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(device)

        # Set up metrics
        self.bleu = BLEUScore()
        self.rouge = ROUGEScore()

        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load test dataset
        self.test_dataset = ArabicQADataset(
            data_path=test_data_path,
            tokenizer=self.tokenizer,
            max_length=32,
            is_training=False
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,  # Process one at a time for detailed analysis
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn
        )

    def generate_answer(self, question: str, max_length: int = 100) -> str:
        """
        Generate an answer for a single question.
        """
        # Encode the question
        inputs = self.tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32
        ).to(self.device)

        # Generate answer
        with torch.no_grad():
            output_sequences = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=4,
                no_repeat_ngram_size=2,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode and return the generated answer
        answer = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        return answer

    def evaluate_model(self, output_file: str = "evaluation_results.json"):
        """
        Evaluate the model on the test set and save results.
        """
        logger.info("Starting model evaluation...")
        results = []
        total_bleu = 0
        total_rouge = 0

        self.model.eval()

        with torch.no_grad():
            for idx in range(len(self.test_dataset)):
                # Get example
                example = self.test_dataset.get_example_text(idx)
                question = example['question']
                true_answer = example['answer']

                # Generate prediction
                predicted_answer = self.generate_answer(question)

                # Calculate metrics
                bleu_score = self.bleu([predicted_answer], [[true_answer]]).item()
                rouge_scores = self.rouge([predicted_answer], [true_answer])
                rouge_score = rouge_scores['rouge1_fmeasure'].item()

                total_bleu += bleu_score
                total_rouge += rouge_score

                # Store result
                result = {
                    'question_id': example['question_id'],
                    'question': question,
                    'true_answer': true_answer,
                    'predicted_answer': predicted_answer,
                    'bleu_score': bleu_score,
                    'rouge_score': rouge_score
                }
                results.append(result)

                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1} examples...")

        # Calculate average metrics
        avg_bleu = total_bleu / len(results)
        avg_rouge = total_rouge / len(results)

        # Add summary metrics
        summary = {
            'average_bleu': avg_bleu,
            'average_rouge': avg_rouge,
            'num_examples': len(results)
        }

        # Save results
        output = {
            'summary': summary,
            'results': results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(f"Evaluation completed. Results saved to {output_file}")
        logger.info(f"Average BLEU score: {avg_bleu:.4f}")
        logger.info(f"Average ROUGE score: {avg_rouge:.4f}")

        return output


def main():
    # Configuration
    model_path = "./models/gpt2_qa_model_epoch_3"  # Adjust to your saved model path
    test_data_path = "data/test-open.json"

    # Initialize evaluator
    evaluator = QAEvaluator(model_path, test_data_path)

    # Run evaluation
    results = evaluator.evaluate_model()

    # Test a few example questions
    example_questions = [
        "ما هو أكبر كوكب في النظام الشمسي؟",
        "من هو مخترع الهاتف؟",
        "متى تأسست المملكة العربية السعودية؟"
    ]

    print("\nTesting example questions:")
    for question in example_questions:
        answer = evaluator.generate_answer(question)
        print(f"\nQ: {question}")
        print(f"A: {answer}")


if __name__ == "__main__":
    main()