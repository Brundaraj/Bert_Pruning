import torch
import torch_pruning as tp
import logging
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments, glue_tasks_num_labels, BertTokenizerFast
from datasets import load_dataset, load_metric
from thop import profile
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import os
# from _future_ import annotations
from pathlib import Path
import numpy as np
from torch.utils.data import ConcatDataset
from transformers import BertTokenizerFast, DataCollatorWithPadding, BertForSequenceClassification, EvalPrediction, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Set up logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments, glue_tasks_num_labels
from datasets import load_dataset, load_metric
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task_name = 'mrpc'

def prepare_datasets(task_name: str, tokenizer: BertTokenizerFast):
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

    raw_datasets = load_dataset('glue', task_name)
    processed_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=raw_datasets['train'].column_names)

    train_dataset = processed_datasets['train']
    validation_datasets = {'validation': processed_datasets['validation']}
    test_dataset = processed_datasets['test']

    return train_dataset, validation_datasets, test_dataset,raw_datasets

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

def evaluate_model(model_name_or_path, task_name, load_best_model_at_end=False):
    # Load the model and the tokenizer
    model = BertForSequenceClassification.from_pretrained(model_name_or_path)
    tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)
    model.to(device)
    # model.save_pretrained(f"./abc")
    # Prepare the dataset
    train_dataset, validation_datasets, test_dataset,raw_datasets = prepare_datasets(task_name, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer)

    # Set the training parameters
    training_args = TrainingArguments(
        output_dir='./output/trainer',
        do_train=False,
        do_eval=True,
        evaluation_strategy='steps',
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

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=validation_datasets['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Evaluate the moddel
    eval_result = trainer.evaluate()
    print(f"Validation dataset Evaluation results \n: {eval_result}")

    sample = raw_datasets['train'][0]

    #Evaluate on the test dataset
    test_result = trainer.predict(test_dataset)
    print(f"Test dataset evaluation results:\n {test_result.metrics}")
    
    sample = raw_datasets['train'][0]
    total_params = sum(p.numel() for p in model.parameters())
    # inputs = tokenizer(sample['sentence1'], sample['sentence2'], return_tensors='pt').to(device)
    # macs, params = profile(model, inputs=(inputs['input_ids'],), verbose=False)
    # flops = 2 * macs
    # model_size = total_params * 4 / (1024 ** 2)
    # print(f"\n  Parameters: {total_params}, FLOPs: {flops}, Size: {model_size:.2f} MB")
    
    # Evaluate the initial model 
evaluate_model("path to initial model", task_name)
evaluate_model("path to structured pruned model", task_name)
evaluate_model("path to unstructured pruned model", task_name)
