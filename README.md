# DSA4213 Assignment 3: Fine-tuning DistilBERT for Question Answering
By Rachel

## Project Overview

This project explores and implements parameter-efficient fine-tuning (PEFT) techniques to adapt a pretrained Transformer model for a downstream Question Answering (QA) task. The goal is to compare the effectiveness and efficiency of two different PEFT strategies as per the assignment requirements.

The project consists of two main parts:
1.  **Model Fine-Tuning:** A Jupyter notebook (`model_comparison.ipynb`) that fine-tunes a `distilbert-base-uncased` model on the SQuAD v1 dataset using two distinct methods: **LoRA** and **(IA)³**.
2.  **Interactive Web Application:** A web app built with **Flask** that allows a user to enter a custom context and question and see the predictions from both fine-tuned models side-by-side for direct comparison.

## Features
- Fine-tuning of `distilbert-base-uncased` on the SQuAD v1 dataset.
- Implementation and comparison of two PEFT methods: LoRA and (IA)³.
- A hyperparameter sweep on a data subset to determine the optimal learning rate for each method.
- Quantitative evaluation using official SQuAD metrics (Exact Match and F1 Score).
- Qualitative evaluation through a user-friendly Flask web application.

## Project Structure
```plaintext
.
├── model_comparison.ipynb     # Main Jupyter Notebook for all training and evaluation.
├── app.py                     # The backend script for the Flask web application.
├── requirements.txt           # A list of required Python packages.
├── clean_notebook.py           # Cleaned the python notebook, with no widget
├── frontend/                  # Contains all frontend files for the web app.
│   ├── static/
│   │   └── css/
│   │       └── styles.css
│   └── templates/
│       └── index.html
│
├── backend/                   # Contains the trained model checkpoints.
│   ├── results_ia3_final/
│   └── results_lora_final/
│
├── IA3_Fine_Tuning_Loss_3_Epochs.png   # (Generated) Loss plot for the (IA)³ model.
├── LoRA_Fine_Tuning_Loss_3_Epochs.png  # (Generated) Loss plot for the LoRA model.
│
└── README.md                  # This README file.
```

## Setup and Installation
Follow these steps to set up the local environment to run the web application.

**1. Clone the Repository**

git clone https://github.com/racxxhel/Assignment3.git

cd Assignment3

**2. Install Dependencies**

pip install -r requirements.txt


How to Run
There are two main components to this project: reproducing the experiments and running the web app.

1. Reproducing the Training Experiments
The entire training and evaluation pipeline is contained in the model_comparison.ipynb notebook.

Important Note: The full training process for both models is computationally intensive. It is strongly recommended to run this notebook in a GPU-accelerated environment like Google Colab, as the training can take over 10 hours on a CPU. 

To run, simply open the notebook and execute the cells from top to bottom. This will:
- Download the SQuAD dataset.
- Perform the hyperparameter sweeps.
- Train the final LoRA and (IA)³ models for 3 epochs.
- Save the trained model files to results_lora_final/ and results_ia3_final/.
- Save the loss plots as PNG images.
- Print the final evaluation scores.

2. Running the Flask Web App
The Flask app allows you to interactively test and compare the two fine-tuned models.

Prerequisite: You must first run the training notebook to generate the model files.

Steps:
1. After running the training notebook, download the final model checkpoint folders
2. Place these checkpoint folders into a new directory named models/ at the root of your project.
3. Update the paths LORA_MODEL_PATH and IA3_MODEL_PATH at the top of the app.py file to point to these checkpoint folders.
4. Run the Flask app from your terminal (make sure your virtual environment is activated)

## Results
Evaluation on the SQuAD v1 validation set revealed that the LoRA fine-tuning strategy substantially outperformed the (IA)³ strategy.
| Model              | Exact Match (EM) | F1 Score |
|--------------------|------------------|-----------|
| **LoRA**           | 62.3841          | 72.8802   |
| **(IA)³**          | 32.3652          | 41.8148   |

## Conclusion:
This project successfully implemented and compared two distinct parameter-efficient fine-tuning (PEFT) methods such as LoRA and (IA)³ for adapting a pretrained DistilBERT model to the task of extractive question answering. The experimental results clearly demonstrate that LoRA was the superior method for this task, achieving a significantly higher F1 score of 72.88 compared to 41.81 from the (IA)³ model. While LoRA delivered better performance, the (IA)³ method was even more parameter-efficient, highlighting a crucial trade-off between predictive accuracy and the number of trainable parameters. The key takeaway is that the choice of PEFT method is not trivial; for this task, an additive method that learns updates to the model's weights (LoRA) proved more effective than a multiplicative method that rescales existing activations ((IA)³). 