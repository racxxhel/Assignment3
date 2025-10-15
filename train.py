import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, IA3Config
import evaluate
import numpy as np
import os
import time
import argparse
import collections
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

def prepare_train_features(examples, tokenizer):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=256,
        stride=64,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples["offset_mapping"]

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples["example_id"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        tokenized_examples["example_id"].append(examples["id"][sample_index])

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])
        token_start_index = 0
        while sequence_ids[token_start_index] != 1: token_start_index += 1
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1: token_end_index -= 1
        if offsets[token_start_index][0] > start_char or offsets[token_end_index][1] < end_char:
            tokenized_examples["start_positions"].append(0)
            tokenized_examples["end_positions"].append(0)
        else:
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char: token_start_index += 1
            start_pos = token_start_index - 1
            while token_end_index >= 0 and offsets[token_end_index][1] >= end_char: token_end_index -= 1
            end_pos = token_end_index + 1
            tokenized_examples["start_positions"].append(start_pos)
            tokenized_examples["end_positions"].append(end_pos)

    return tokenized_examples

def plot_loss_curves(log_history, save_path, title):
    """
    Plots training and validation loss curves from the Trainer's log history.
    """
    train_logs = [log for log in log_history if 'loss' in log]
    val_logs = [log for log in log_history if 'eval_loss' in log]

    train_epochs = [log['epoch'] for log in train_logs]
    train_losses = [log['loss'] for log in train_logs]
    val_epochs = [log['epoch'] for log in val_logs]
    val_losses = [log['eval_loss'] for log in val_logs]

    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_losses, label='Training Loss')
    plt.plot(val_epochs, val_losses, label='Validation Loss', marker='o')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

    plt.show()

def postprocess_qa_predictions(examples, features, raw_predictions, tokenizer, n_best_size=20, max_answer_length=30):
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = collections.OrderedDict()

    print("Post-processing predictions...")
    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]
        min_null_score = None
        valid_answers = []
        context = example["context"]
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {"score": start_logits[start_index] + end_logits[end_index], "text": context[start_char:end_char]}
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}

        predictions[example["id"]] = best_answer["text"]

    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return formatted_predictions, references

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, required=True, choices=["lora", "ia3"],
        help="The PEFT method to use for training."
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size.")
    return parser.parse_args()

def main():
    args = get_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using Device: {device}')
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    print("Loading and preprocessing SQuAD dataset...")
    dataset = load_dataset("squad")
    
    train_dataset = dataset["train"].map(
        lambda examples: prepare_train_features(examples, tokenizer),
        batched=True, remove_columns=dataset["train"].column_names
    )
    validation_dataset = dataset["validation"].map(
        lambda examples: prepare_train_features(examples, tokenizer),
        batched=True, remove_columns=dataset["validation"].column_names
    )
    
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    if args.model_type == "lora":
        peft_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.1, task_type=TaskType.QUESTION_ANS, target_modules=["q_lin", "v_lin"]
        )
    elif args.model_type == "ia3":
        peft_config = IA3Config(
            task_type=TaskType.QUESTION_ANS,
            target_modules=["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"],
            feedforward_modules=["lin1", "lin2"],
        )
    
    peft_model = get_peft_model(model, peft_config).to(device)
    print(f"\nTraining with {args.model_type.upper()} configuration:")
    peft_model.print_trainable_parameters()
    
    output_dir = os.path.join("backend", f"results_{args.model_type}_final")
    plot_save_path = os.path.join("backend", f"{args.model_type.upper()}_Fine_Tuning_Loss.png")
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=500,
        report_to="none"
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )

    print("\nStarting training...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    print(f"Total Training Time: {(end_time - start_time) / 3600:.2f} hours")
    
    plot_loss_curves(
        trainer.state.log_history,
        plot_save_path,
        f"{args.model_type.upper()} Training/Validation Loss ({args.epochs} Epochs)"
    )

    print("\nStarting final evaluation on validation set...")
    squad_metric = evaluate.load("squad")
    raw_predictions = trainer.predict(validation_dataset)
    
    final_predictions, final_references = postprocess_qa_predictions(
        dataset["validation"], 
        validation_dataset, 
        raw_predictions.predictions, 
        tokenizer=tokenizer
    )
    
    metrics = squad_metric.compute(predictions=final_predictions, references=final_references)
    print(f"\n{args.model_type.upper()} Model Final Evaluation Report:")
    print(f"  Exact Match (EM): {metrics['exact_match']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")

if __name__ == "__main__":
    main()