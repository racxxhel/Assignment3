# DSA4213 Assignment 3: Fine-tuning DistilBERT for Question Answering
By Tan Hwee Li Rachel

## Project Overview

This project explores and implements parameter-efficient fine-tuning (PEFT) techniques to adapt a pretrained Transformer model for a downstream Question Answering (QA) task. The goal is to compare the effectiveness and efficiency of two different PEFT strategies as per the assignment requirements.

The project is organized into three main components:

***Reproducible Training Script*** (`train.py`): This is the primary script for fine-tuning and evaluation. It is designed to be run from the command line to easily reproduce the experimental results reported.

***Interactive Web Application*** (`app.py`): A web app built with Flask that allows a user to perform qualitative analysis by entering a custom context and question and seeing side-by-side predictions from both fine-tuned models.

***Development Notebook*** (`model_comparison.ipynb`): This Jupyter Notebook provides a detailed, step-by-step documentation of the original development and experimentation process. It was used to generate the findings for the written report.

## Features
- Fine-tuning of `distilbert-base-uncased` on the SQuAD v1 dataset.
- Implementation and comparison of two PEFT methods: LoRA and (IA)³.
- A hyperparameter sweep on a data subset to determine the optimal learning rate for each method.
- Quantitative evaluation using official SQuAD metrics (Exact Match and F1 Score).
- Qualitative evaluation through a user-friendly Flask web application.

## Project Structure
```plaintext
.
├── train.py                   # Main script for training and evaluation for reproducibility.
├── app.py                     # Backend script for the Flask web application.
├── requirements.txt           # A list of required Python packages.
├── .gitignore                 # Specifies files for Git to ignore.
├── frontend/                  # Contains all frontend files for the web app.
│   ├── static/
│   │   └── css/
│   │       └── styles.css
│   └── templates/
│       └── index.html
├── backend/                   # Directory for generated model checkpoints and plots.
│   └── (This folder will be populated by train.py)
├── model_comparison.ipynb     # Inference notebook documenting the development process for report.
└── evaluation_examples.ipynb     # Qualitative evaluation notebook used for report.
└── README.md                  # This README file.
└── Application_Example_photo_1.png    # Example photo of the Web App
└── Application_Example_photo_2.png    # 2nd Example photo of the Web App
└── Report Analysis     # Report on the experiment 
```

## Setup and Installation
Follow these steps to set up the local environment to run the web application.

**1. Clone the Repository**
```bash
git clone https://github.com/racxxhel/Assignment3.git

cd Assignment3
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

## How to Run:

There are two main components to this project: reproducing the experiments and running the web app.

**1. Reproducing the Training Experiments**

The entire training and evaluation pipeline is handled by the `train.py` script.

**Important Note:** The full training process is computationally intensive. It is strongly recommended to run this script in a GPU-accelerated environment (e.g., Google Colab, Kaggle).

To train a model, run the script from your terminal with the desired model type:
## To train and evaluate the LoRA model for 3 epochs
```bash
python train.py --model_type lora --epochs 3
```

## To train and evaluate the (IA)³ model for 3 epochs
```bash
python train.py --model_type ia3 --epochs 3
```

This script will automatically:
- Download the SQuAD dataset.
- Train the final LoRA and (IA)³ models for 3 epochs using the optimal hyperparameter.
- Save the trained model files to backend directory.
- Print the final evaluation scores.

**2. Running the Flask Web App**

The Flask app allows you to interactively test and compare the two fine-tuned models.

**Prerequisite**: You must first run the `train.py` script for both lora and ia3 models to generate the required model files in the `backend/` directory.

Once models are trained, run the app from your terminal: 
```bash
python app.py
```

## Results
Evaluation on the SQuAD v1 validation set revealed that the LoRA fine-tuning strategy substantially outperformed the (IA)³ strategy.
| Model              | Exact Match (EM) | F1 Score |
|--------------------|------------------|-----------|
| **LoRA**           | 62.3841          | 72.8802   |
| **(IA)³**          | 32.3652          | 41.8148   |

## Application Demonstration

To provide a practical and interactive way to compare the performance of the two fine-tuned models, a simple web application was developed using Flask.

The application allows a user to input any custom context and question. Upon submission, it runs inference with both the LoRA-tuned model and the (IA)³-tuned model and displays their predicted answers side-by-side. Furthermore, the user can provide an optional "True Answer" to see a real-time calculation of the Exact Match (EM) and F1 Score for each model's prediction, offering immediate quantitative feedback on their performance.This provides a direct method for the qualitative "error analysis" and comparison of strategies required by the assignment.

#### Example Usage:
The user provides a context about the history of NUS and asks a specific question about a date.

![User interface for inputting context, question and optional True Answer](./Application_Example_photo_1.png)

The application displays the predictions from both models. In this example, the LoRA model correctly extracts the full date ("8 october 1949"), achieving a perfect F1 score. In contrast, the (IA)³ model extracts only a partial answer ("1949"), resulting in a lower score. This demonstrates the app's utility in highlighting the nuanced performance differences between the two fine-tuning methods.

![Side-by-side comparison of LoRA and (IA)³ model outputs](./Application_Example_photo_2.png)

## Conclusion:
This project successfully implemented and compared two distinct parameter-efficient fine-tuning (PEFT) methods such as LoRA and (IA)³ for adapting a pretrained DistilBERT model to the task of extractive question answering. The experimental results clearly demonstrate that LoRA was the superior method for this task, achieving a significantly higher F1 score of 72.88 compared to 41.81 from the (IA)³ model. While LoRA delivered better performance, the (IA)³ method was even more parameter-efficient, highlighting a crucial trade-off between predictive accuracy and the number of trainable parameters. The key takeaway is that the choice of PEFT method is not trivial, for this task, an additive method that learns updates to the model's weights (LoRA) proved more effective than a multiplicative method that rescales existing activations (IA)³.