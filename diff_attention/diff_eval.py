import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader
from diff_dataset import ArabicQADataset
from torchmetrics.text import BLEUScore
import json
import logging
import os
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
            self,
            model_path: str,
            test_data_path: str,
            batch_size: int = 8,
            max_length: int = 128,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.max_length = max_length

        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}")
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Prepare dataset and dataloader
        self.test_dataset = ArabicQADataset(
            data_path=test_data_path,
            tokenizer=self.tokenizer,
            max_length=max_length,
            is_training=False
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn
        )

        # Initialize BLEU scorer
        self.bleu = BLEUScore()

    def generate_answer(self, question: str) -> str:
        """Generate an answer for a given question."""
        input_text = f"Question: {question} Answer:"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=self.max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_answer = generated_answer.replace(input_text, "").strip()

        return generated_answer

    def evaluate(self) -> dict:
        """Evaluate the model on the test dataset."""
        self.model.eval()
        generated_answers = []
        reference_answers = []

        logger.info("Starting evaluation...")
        for batch in tqdm(self.test_loader):
            questions = [self.test_dataset.data[i]['question'] for i in range(len(batch['input_ids']))]
            answers = [self.test_dataset.data[i]['answer'] for i in range(len(batch['input_ids']))]

            for question, ref_answer in zip(questions, answers):
                generated_answer = self.generate_answer(question)
                generated_answers.append(generated_answer.split())
                reference_answers.append([ref_answer.split()])

        # Calculate BLEU score
        bleu_score = self.bleu(generated_answers, reference_answers)

        return {'bleu_score': float(bleu_score)}


def main():
    # Configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    configs = {
        'test_data_path': os.path.join(parent_dir, 'data', 'test-open.json'),
        'batch_size': 8,
        'max_length': 128,
        'results_dir': os.path.join(current_dir, 'evaluation_results')
    }

    # Create results directory
    os.makedirs(configs['results_dir'], exist_ok=True)

    # Evaluate models for different ratios
    ratios = [0.25, 0.35, 0.5]
    results = {}

    for ratio in ratios:
        logger.info(f"\nEvaluating model with ratio {ratio}")
        # Updated model path to match your project structure
        model_path = os.path.join(current_dir, 'diff_models', f'ratio_{ratio}_ratio_{ratio}_epoch_3')

        try:
            evaluator = Evaluator(
                model_path=model_path,
                test_data_path=configs['test_data_path'],
                batch_size=configs['batch_size'],
                max_length=configs['max_length']
            )

            metrics = evaluator.evaluate()
            results[f"ratio_{ratio}"] = metrics

            logger.info(f"BLEU score for ratio {ratio}: {metrics['bleu_score']:.4f}")

        except Exception as e:
            logger.error(f"Error evaluating model with ratio {ratio}: {str(e)}")
            continue

    # Save results
    results_path = os.path.join(configs['results_dir'], 'bleu_scores.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    logger.info(f"\nEvaluation completed! Results saved to {results_path}")


if __name__ == "__main__":
    main()