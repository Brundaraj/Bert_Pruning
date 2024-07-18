# Bert_Pruning
This project demonstrates the process of fine-tuning and pruning a BERT model using the GLUE benchmark. The primary goals are to fine-tune the model on the dataset, evaluate its performance, and prune the model to reduce its size and computational complexity.

# Prerequisites
Python 3.8+
PyTorch 1.8+
Transformers library by Hugging Face
Datasets library by Hugging Face
THOP library
NumPy
Scikit-learn

#Installation
Clone the repository:
git clone https://github.com/Brundaraj/bert-finetuning-pruning.git

Install the required packages:
pip install -r requirements.txt

# Usage

# Structured Pruning
Perform structured pruning:
python Structured_pruning.py

This script will prune the fine-tuned BERT model, save the pruned model, and print the evaluation metrics for the pruned model.

# Unstructured Pruning
Perform unstructured pruning:
python Unstructured_pruning.py

This script will perform unstructured pruning on the fine-tuned BERT model, save the pruned model, and print the evaluation metrics.

#Demo Evaluation
Evaluate the pruned model:
python demo_eval.py

This script will evaluate the pruned BERT model on the MRPC dataset and print the evaluation metrics.

