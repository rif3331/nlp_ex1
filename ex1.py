import argparse
import os
import numpy as np
import wandb
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

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

    model_name = "bert-base-uncased"
    output_dir = "/content/gdrive/MyDrive/anlp_ex1"

    os.makedirs(output_dir, exist_ok=True)

    dataset = load_dataset("glue", "mrpc")

    model_path = args.model_path if args.model_path is not None else model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if args.max_train_samples != -1:
        dataset["train"] = dataset["train"].select(range(args.max_train_samples))
    if args.max_eval_samples != -1:
        dataset["validation"] = dataset["validation"].select(range(args.max_eval_samples))
    if args.max_predict_samples != -1:
        dataset["test"] = dataset["test"].select(range(args.max_predict_samples))

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

    run_name = f"bert_mrpc_epochs_{args.num_train_epochs}_lr_{args.lr}_bs_{args.batch_size}"

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, predictions)}

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, run_name),
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        logging_steps=1,
        report_to="wandb" if args.do_train else "none",
        run_name=run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if args.do_train:
        wandb.init(
            project="anlp_ex1",
            name=run_name,
            config={
                "model": model_name,
                "epochs": args.num_train_epochs,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
            }
        )

        trainer.train()
        eval_results = trainer.evaluate()
        print(eval_results)

        final_model_path = os.path.join(output_dir, run_name, "final_model")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)

        wandb.finish()

    if args.do_predict:
        predictions_output = trainer.predict(tokenized_dataset["test"])
        predictions = np.argmax(predictions_output.predictions, axis=-1)

        with open("predictions.txt", "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(str(pred) + "\n")

        print("Saved predictions to predictions.txt")

if __name__ == "__main__":
    main()