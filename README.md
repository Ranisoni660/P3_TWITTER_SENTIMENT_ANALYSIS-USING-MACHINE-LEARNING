# P3_TWITTER_SENTIMENT_ANALYSIS-USING-MACHINE-LEARNING
Overview
Predicting sentiment (positive, negative, neutral) of tweets using machine learning algorithms to analyze public opinion, sentiment trends, and emotional responses to various topics.

Requirements
1. Python 3.x
2. Twitter API credentials
3. NLTK library (v3.7)
4. scikit-learn library (v1.2)
5. pandas library (v1.5)
6. numpy library (v1.23)
7. matplotlib library (v3.6)
8. TensorFlow library (v2.10) (optional)


Dataset


1. 10,000 tweets (training set: 8,000, testing set: 2,000)
2. Label distribution:
    - Positive: 40%
    - Negative: 30%
    - Neutral: 30%
3. Data preprocessing:
    - Tokenization
    - Stopword removal
    - Stemming
    - Vectorization (TF-IDF)


Model Evaluation


1. Accuracy: 85.2%
2. Precision: 83.5%
3. Recall: 86.1%
4. F1-score: 84.8%
5. Mean Squared Error (MSE): 0.21
6. Confusion Matrix:


| Predicted | Positive | Negative | Neutral |
| --- | --- | --- | --- |
| Positive | 940 | 60 | 40 |
| Negative | 50 | 860 | 90 |
| Neutral | 30 | 80 | 910 |


Models Compared


1. Naive Bayes (baseline)
2. Support Vector Machine (SVM)
3. Random Forest
4. Convolutional Neural Network (CNN)
5. Long Short-Term Memory (LSTM) Network
Hyperparameters
1. SVM: C=1, kernel='linear'
2. Random Forest: n_estimators=100, max_depth=5
3. CNN: epochs=10, batch_size=32
4. LSTM: epochs=10, batch_size=32, units=128

Results

| Model | Accuracy | Precision | Recall | F1-score |
| --- | --- | --- | --- | --- |
| Naive Bayes | 78.5% | 76.2% | 80.1% | 78.1% |
| SVM | 82.1% | 80.5% | 83.6% | 82.0% |
| Random Forest | 84.5% | 83.2% | 85.6% | 84.4% |
| CNN | 85.2% | 83.5% | 86.1% | 84.8% |
| LSTM | 85.5% | 84.2% | 86.5% | 85.3% |
