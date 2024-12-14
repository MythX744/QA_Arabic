from transformers import AutoTokenizer
from dataset import ArabicQADataset
from torch.utils.data import DataLoader


def test_dataset():
    """Test the dataset implementation."""
    print("Starting dataset test...")

    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/aragpt2-base")

    # Set padding token
    print("Setting up padding token...")
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataset instance
    print("Creating dataset instance...")
    dataset = ArabicQADataset(
        data_path='data/train-open.json',
        tokenizer=tokenizer,
        max_length=512,
        is_training=True
    )

    print(f"Dataset size: {len(dataset)}")

    # Test single item
    print("\nTesting single item retrieval...")
    item = dataset[0]
    print("Keys in item:", item.keys())
    assert all(k in item for k in ['input_ids', 'attention_mask', 'labels'])
    print("Single item test passed!")

    # Show example
    print("\nExample from dataset:")
    example_text = dataset.get_example_text(0)
    print("Question:", example_text['question'])
    print("Answer:", example_text['answer'])

    # Test batch creation
    print("\nTesting batch creation...")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )

    batch = next(iter(dataloader))
    print("Keys in batch:", batch.keys())
    print("Batch shapes:")
    for k, v in batch.items():
        if v is not None:
            print(f"{k}: {v.shape}")

    assert all(k in batch for k in ['input_ids', 'attention_mask', 'labels'])
    print("Batch creation test passed!")

    print("\nAll dataset tests passed successfully!")


if __name__ == "__main__":
    test_dataset()