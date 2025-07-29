BERT Text Classification
This project implements a BERT-based Natural Language Processing (NLP) model to classify text into predefined categories.
It includes data preprocessing, fine‑tuning of the BERT model, and evaluation to achieve high‑accuracy predictions.

Technologies Used:
- Python 3.8+
- Hugging Face Transformers
- PyTorch
- Pandas / NumPy
- Scikit-learn

How It Works:
- Loads and preprocesses the dataset (tokenization, padding, attention masks).
- Fine‑tunes the BERT model on the training data.
- Evaluates the model on the test data using accuracy, precision, recall, and F1‑score.

Usage:
- Install dependencies: pip install -r requirements.txt
- Prepare your dataset as CSV files with:
  - text column for input text
  - label column for target categories
- Run the script: python code_for_BERT.py

Results:
On the dataset used, the model achieved:
  - Accuracy: XX%
  - F1‑score: XX%
  (Replace with your actual results.)

Notes:
- Dataset is not included due to licensing restrictions.
- You can use any CSV dataset with the same format.
- For faster training, use a GPU.
