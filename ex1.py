import argparse
from datasets import load_dataset
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_predict_samples", type=int, default=-1)

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")

    parser.add_argument("--model_path", type=str, default=None)

    args = parser.parse_args()

    dataset = load_dataset("glue", "mrpc")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    print(tokenized_dataset)

if __name__ == "__main__":
    main()