# Spam Detection Project

## Overview
This project implements a machine learning-based spam email detector using the **Enron Email Dataset**. The model classifies messages as either **spam** or **ham (not spam)** based on their content. It leverages **text preprocessing**, **TF-IDF vectorization**, and **Multinomial Naive Bayes** for accurate predictions.  
The goal of this project is to showcase a complete workflow of a text classification task, including data preprocessing, model training, evaluation, and deployment-ready prediction functions.

---

## Dataset
- The dataset used is the **Enron Email Dataset**.  
- **Location in repo:** `enron_spam_data.zip`  
- After downloading or cloning the repository, unzip the dataset. You should have a CSV file, e.g., `enron_spam_data.csv`.  

**Instructions to use the dataset:**
- The code rely on this CSV file for training and prediction. 

**Original source:** [Kaggle - Enron Email Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)  

---

## Project Structure
spam-detection/
│── README.md with instructions
│── .ipynb file
│── requirements.txt # Python dependencies
│── README.md
│── dataset.csv

yaml
Copy code

---

## Installation
1. Clone the repository:

```bash
git clone https://github.com/your-username/spam-detection.git
cd spam-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```
Unzip the dataset in the data/ folder. Ensure the CSV file is at:
data/enron_spam_data.csv

## Usage
1. Train the Model
```bash
python src/train.py
```
- Trains the Naive Bayes model on the dataset.
- Saves the trained model (spam_classifier.pkl) and vectorizer (vectorizer.pkl) in models/.
- Generates evaluation metrics (metrics.txt) and confusion matrix (confusion_matrix.png) in results/.

2. Predict New Messages
```bash
python src/predict.py
```
- Prompts for a message input and outputs whether it is Spam or Ham.

## Results
- Accuracy: 98.67%
- Precision: 98.58%
- Recall: 98.67%
- F1-score: 98.63%

## Contributing
Feel free to fork the project, add improvements, or experiment with other models.
