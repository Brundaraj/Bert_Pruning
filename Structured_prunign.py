import torch
import torch_pruning as tp
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments, glue_tasks_num_labels, BertTokenizerFast
from datasets import load_dataset, load_metric
from thop import profile
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import os
from __future__ import annotations
from pathlib import Path
import numpy as np
from torch.utils.data import ConcatDataset
from transformers import BertTokenizerFast, DataCollatorWithPadding, BertForSequenceClassification, EvalPrediction, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
os.environ["TOKENIZERS_PARALLELISM"] = "false"

task_name = 'mrpc'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(pretrained_model_name_or_path: str, task_name: str):
    is_regression = task_name == 'stsb'
    num_labels = 1 if is_regression else (3 if task_name == 'mnli' else 2)
    model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, num_labels=num_labels)
    return model

def prepare_datasets(task_name: str, tokenizer: BertTokenizerFast, cache_dir: str):
    task_to_keys = {
        'cola': ('sentence', None),
        'mnli': ('premise', 'hypothesis'),
        'mrpc': ('sentence1', 'sentence2'),
        'qnli': ('question', 'sentence'),
        'qqp': ('question1', 'question2'),
        'rte': ('sentence1', 'sentence2'),
        'sst2': ('sentence', None),
        'stsb': ('sentence1', 'sentence2'),
        'wnli': ('sentence1', 'sentence2'),
    }
    sentence1_key, sentence2_key = task_to_keys[task_name]

    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=False, max_length=128, truncation=True)

        if 'label' in examples:
            result['labels'] = examples['label']
        return result

    raw_datasets = load_dataset('glue', task_name, cache_dir=cache_dir)
    processed_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=raw_datasets['train'].column_names)

    train_dataset = processed_datasets['train']
    if task_name == 'mnli':
        validation_datasets = {
            'validation_matched': processed_datasets['validation_matched'],
            'validation_mismatched': processed_datasets['validation_mismatched']
        }
    else:
        validation_datasets = {
            'validation': processed_datasets['validation']
        }

    # Load the test dataset
    test_dataset = processed_datasets['test']

    return train_dataset, validation_datasets, test_dataset

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    mcc = matthews_corrcoef(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'mcc': mcc,
    }

def prepare_traced_trainer(model, task_name, load_best_model_at_end=False):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    train_dataset, validation_datasets, test_dataset = prepare_datasets(task_name, tokenizer, None)
    data_collator = DataCollatorWithPadding(tokenizer)
    training_args = TrainingArguments(
        output_dir='./output/trainer',
        do_train=True,
        do_eval=True,
        evaluation_strategy='steps',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        dataloader_num_workers=8,
        learning_rate=3e-5,
        save_strategy='steps',
        save_total_limit=1,
        metric_for_best_model='default',
        load_best_model_at_end=load_best_model_at_end,
        disable_tqdm=True,
        optim='adamw_torch',
        seed=1024
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_datasets['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    return trainer, tokenizer, test_dataset

def build_finetuning_model(task_name: str, state_dict_path: str):
    model = build_model('bert-base-uncased', task_name)
    if Path(state_dict_path).exists():
        model.load_state_dict(torch.load(state_dict_path))
    else:
        trainer, _, _ = prepare_traced_trainer(model, task_name, True)
        trainer.train()
        torch.save(model.state_dict(), state_dict_path)
    return model

# Load and fine-tune the model
model = build_finetuning_model(task_name, "./model_state_dict.pth")
model.to(device)

# Evaluate initial model
metric = load_metric("glue", task_name)
trainer, tokenizer, test_dataset = prepare_traced_trainer(model, task_name, True)
initial_eval_result = trainer.evaluate()
print("Initial model evaluation results:", initial_eval_result)


# Save initial model
initial_model_dir = "./initial_model"
model.save_pretrained(initial_model_dir)
tokenizer.save_pretrained(initial_model_dir)
if not os.path.exists(f"{initial_model_dir}/pytorch_model.bin"):
    torch.save(model.state_dict(), f"{initial_model_dir}/pytorch_model.bin")

# Evaluate on test dataset
test_result = trainer.predict(test_dataset)
print("Initial model Test dataset evaluation results:", test_result.metrics)

# Move model to GPU
model.to(device)

raw_datasets = load_dataset('glue', task_name)
sample = raw_datasets['train'][0]

def compute_model_size_and_flops(model, sample, tokenizer, device):
    total_params = sum(p.numel() for p in model.parameters())
    inputs = tokenizer(sample['sentence1'], sample['sentence2'], return_tensors='pt').to(device)
    macs, params = profile(model, inputs=(inputs['input_ids'],), verbose=False)
    flops = 2 * macs
    model_size = total_params * 4 / (1024 ** 2)
    return total_params, flops, model_size

initial_total_params, initial_flops, initial_model_size = compute_model_size_and_flops(model, sample, tokenizer, device)
print(f"Initial model - Parameters: {initial_total_params}, FLOPs: {initial_flops}, Size: {initial_model_size:.2f} MB")
print(f"Initial number of layers: {len(model.bert.encoder.layer)}")
print(f"Accuracy - Initial: {initial_eval_result['eval_accuracy']}")
print(f"F1 Score - Initial: {initial_eval_result['eval_f1']}")
print(f"MCC - Initial: {initial_eval_result['eval_mcc']}")


# Prune model layers
def prune_layers(model, prune_ratio=0.4):
    num_layers = len(model.bert.encoder.layer)
    num_layers_to_prune = int(num_layers * prune_ratio)
    new_encoder_layers = [model.bert.encoder.layer[i] for i in range(num_layers - num_layers_to_prune)]
    model.bert.encoder.layer = torch.nn.ModuleList(new_encoder_layers)

prune_layers(model, prune_ratio=0.4)

# Save pruned model
pruned_model_dir = "./pruned_model"
model.save_pretrained(pruned_model_dir)
tokenizer.save_pretrained(pruned_model_dir)
if not os.path.exists(f"{pruned_model_dir}/pytorch_model.bin"):
    torch.save(model.state_dict(), f"{pruned_model_dir}/pytorch_model.bin")

# Load pruned model and evaluate
#pruned_model = build_model(pruned_model_dir, task_name)
#pruned_model.load_state_dict(torch.load(f"{pruned_model_dir}/pytorch_model.bin", map_location=device))
pruned_model=model

pruned_model.to(device)
trainer,tokenizer,_ = prepare_traced_trainer(pruned_model, task_name, True)
trainer.train()


pruned_eval_result = trainer.evaluate()
print("Pruned model evaluation results:", pruned_eval_result)


pruned_total_params, pruned_flops, pruned_model_size = compute_model_size_and_flops(pruned_model, sample, tokenizer, device)
print(f"Pruned model - Parameters: {pruned_total_params}, FLOPs: {pruned_flops}, Size: {pruned_model_size:.2f} MB")
print(f"Pruned number of layers: {len(pruned_model.bert.encoder.layer)}")
print("Comparison of Initial and Pruned Models:")
print(f"Accuracy - Pruned: {pruned_eval_result['eval_accuracy']}")
print(f"F1 Score -  Pruned: {pruned_eval_result['eval_f1']}")
print(f"MCC -  Pruned: {pruned_eval_result['eval_mcc']}")

# Evaluate pruned model on test dataset
pruned_test_result = trainer.predict(test_dataset)
print("Pruned model test dataset evaluation results:", pruned_test_result.metrics)

