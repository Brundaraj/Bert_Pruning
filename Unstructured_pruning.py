# Import necessary libraries
import os
import torch
import numpy as np
import pandas as pd
from thop import profile
from pathlib import Path
import torch_pruning as tp
from __future__ import annotations
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import load_dataset, load_metric
from transformers.trainer_callback import TrainerCallback
from transformers import TrainerCallback, TrainerState, TrainerControl
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from transformers import  DataCollatorWithPadding,  EvalPrediction, glue_tasks_num_labels, BertTokenizer


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
        eval_strategy='steps',  # Updated to eval_strategy
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        dataloader_num_workers=2,  # Reduced to avoid multiprocessing issues
        learning_rate=3e-5,
        save_strategy='steps',
        save_total_limit=1,
        metric_for_best_model='accuracy',  # Ensure this matches a key in compute_metrics
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
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    return trainer, tokenizer, test_dataset,train_dataset

def build_finetuning_model(task_name: str, state_dict_path: str):
    model = build_model('bert-base-uncased', task_name)
    if Path(state_dict_path).exists():
        model.load_state_dict(torch.load(state_dict_path))
    else:
        trainer, _, _,train_dataset = prepare_traced_trainer(model, task_name, True)
        trainer.train()
        torch.save(model.state_dict(), state_dict_path)
    return model

# Load and fine-tune the model
model = build_finetuning_model(task_name, "./model_state_dict.pth")
model.to(device)


tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# Save initial model
initial_model_dir = "./initial_model"
model.save_pretrained(initial_model_dir)
tokenizer.save_pretrained(initial_model_dir)
if not os.path.exists(f"{initial_model_dir}/pytorch_model.bin"):
    torch.save(model.state_dict(), f"{initial_model_dir}/pytorch_model.bin")


# Evaluate initial model
metric = load_metric("glue", task_name)
trainer, tokenizer, test_dataset,train_dataset = prepare_traced_trainer(model, task_name, True)
initial_eval_result = trainer.evaluate()
print("Initial model evaluation results:", initial_eval_result)

raw_datasets = load_dataset('glue', task_name)
sample = raw_datasets['train'][0]

def compute_model_size_and_flops(model, sample, tokenizer, device):
    total_params = sum(p.numel() for p in model.parameters())
    
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
    if sentence2_key is None:
        inputs = tokenizer(sample[sentence1_key], return_tensors='pt').to(device)
    else:
        inputs = tokenizer(sample[sentence1_key], sample[sentence2_key], return_tensors='pt').to(device)
        
    macs, params = profile(model, inputs=(inputs['input_ids'],), verbose=False)
    flops = 2 * macs
    model_size = total_params * 4 / (1024 ** 2)
    return total_params, flops, model_size

initial_total_params, initial_flops, initial_model_size = compute_model_size_and_flops(model, sample, tokenizer, device)
print(f"Initial model - Parameters: {initial_total_params}, FLOPs: {initial_flops}, Size: {initial_model_size:.2f} MB")
print(f"Initial number of layers: {len(model.bert.encoder.layer)}")

# The following lines are commented out because `initial_eval_result` is not defined.
# You would need to include an evaluation step before printing these results.
# print(f"Accuracy - Initial: {initial_eval_result['eval_accuracy']}")
# print(f"F1 Score - Initial: {initial_eval_result['eval_f1']}")
# print(f"MCC - Initial: {initial_eval_result['eval_mcc']}")

# Function to compute gradients
def compute_gradients(model, dataloader, criterion, device):
    model.train()
    model.to(device)
    
    gradients = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}
    
    for batch in dataloader:
        inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
        labels = batch['labels'].to(device)
        
        model.zero_grad()
        outputs = model(**inputs)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                gradients[name] += param.grad.data.clone()
    
    return gradients

# Function for gradient-based pruning
def gradient_based_pruning(model, gradients, pruning_percentage=0.4):
    all_gradients = torch.cat([grad.view(-1) for grad in gradients.values()])
    threshold = torch.sort(all_gradients.abs())[0][int(pruning_percentage * len(all_gradients))]
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            mask = gradients[name].abs() > threshold
            param.data.mul_(mask.float())
    
    return model

data_collator = DataCollatorWithPadding(tokenizer)
dataloader = DataLoader(train_dataset, batch_size=32, collate_fn=data_collator)
criterion = torch.nn.CrossEntropyLoss()

# Compute gradients
gradients = compute_gradients(model, dataloader, criterion, device)
pruning_percentage = 0.4  # Prune 40% of the weights
pruned_model = gradient_based_pruning(model, gradients, pruning_percentage)


# Save pruned model
pruned_model_dir = "./pruned_model"
model.save_pretrained(pruned_model_dir)
tokenizer.save_pretrained(pruned_model_dir)
if not os.path.exists(f"{pruned_model_dir}/pytorch_model.bin"):
    torch.save(model.state_dict(), f"{pruned_model_dir}/pytorch_model.bin")

pruned_model=model

pruned_model.to(device)
trainer,tokenizer,_,train_dataset = prepare_traced_trainer(pruned_model, task_name, True)
trainer.train()

pruned_eval_result = trainer.evaluate()
print("Pruned model evaluation results:", pruned_eval_result)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to check sparsity
def check_sparsity(model):
    total_params = 0
    total_zeros = 0
    for param in model.parameters():
        total_params += param.numel()
        total_zeros += torch.sum(param == 0).item()
    sparsity = total_zeros / total_params
    return sparsity, total_params, total_zeros
pruned_params = count_parameters(model)
pruned_sparsity, pruned_total_params, pruned_total_zeros = check_sparsity(model)
pruned_total_params=pruned_params-pruned_total_zeros
print(f"Pruned model sparsity: {pruned_sparsity * 100:.2f}%")
print(f"Pruned total parameters: {pruned_total_params}")
print(f"Pruned zero parameters: {pruned_total_zeros}")

print(f"Pruned number of layers: {len(pruned_model.bert.encoder.layer)}")
print("Comparison of Initial and Pruned Models:")
print(f"Accuracy - Pruned: {pruned_eval_result['eval_accuracy']}")
print(f"F1 Score -  Pruned: {pruned_eval_result['eval_f1']}")
print(f"MCC -  Pruned: {pruned_eval_result['eval_mcc']}")
