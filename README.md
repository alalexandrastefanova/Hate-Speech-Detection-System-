# Hate Speech Detection Using Machine Learning and Deep Learning

This project compares classical machine learning and deep learning approaches for hate speech detection using the Kaggle Hate Speech and Offensive Language dataset.

## Models
- Logistic Regression
- Naive Bayes
- Random Forest
- SVM
- Bidirectional LSTM
- DistilBERT

## Techniques
- TF-IDF
- Bag-of-Words
- NLP preprocessing
- Confusion matrices
- F1 Macro evaluation

## Best Result
DistilBERT:
Accuracy: 0.9161
F1 Macro: 0.7438

## Technologies
Python, Scikit-learn, PyTorch, Transformers, Pandas, Matplotlib

## Repository structure

```text
Hate-Speech-Detection-System/
│
├── images/
│   ├── result_number_comparison.JPG
│   └── results_charts_comparison.png
│
├── notebook/
│   ├── hate_speech_detection_executed.ipynb
│   └── hate_speech_detection_executed_lstm_bert.ipynb
│
├── thesis/
│   └── Bergkvist & Stefanova (2026) Hate Speech Detection Using Machine Learning.pdf
│
├── README.md
└── requirements.txt
```
## Results

### Model Performance Comparison

![Model Results](images/model_performance_table)

Table comparing the performance of traditional machine learning models (Logistic Regression, Naive Bayes, Random Forest, and SVM) and deep learning approaches (Bidirectional LSTM and DistilBERT) using Accuracy, Macro F1, and Weighted F1 metrics.

### Performance Visualizations

![Performance Charts](images/model_performance_charts)

Comparison of Accuracy, Macro F1, and Weighted F1 scores across different feature representations (Bag-of-Words and TF-IDF) and deep learning architectures.
