<p align="center">
  <img src="https://img.shields.io/badge/Sentiment%20Analysis-Consumer%20MultiSource-blueviolet?style=for-the-badge&logo=python&logoColor=white" />
</p>

<h1 align="center">🔍🤖 Consumer Sentiment & Emotion Analysis (Amazon+Twitter+News)</h1>

<p align="center">
  <b>CodeAlpha Internship — Task 4</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Pandas-dataframe-lightgrey?style=for-the-badge&logo=pandas" />
  <img src="https://img.shields.io/badge/NumPy-numeric-yellow?style=for-the-badge&logo=numpy" />
  <img src="https://img.shields.io/badge/ML-Scikit--Learn-orange?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/NLP-NLTK-lightblue?style=for-the-badge&logo=nltk" />
  <img src="https://img.shields.io/badge/Sentiment-VADER-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Sentiment-TextBlob-pink?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Visualization-Matplotlib-green?style=for-the-badge&logo=matplotlib" />
  <img src="https://img.shields.io/badge/Visualization-Seaborn-red?style=for-the-badge&logo=seaborn" />
  <img src="https://img.shields.io/badge/Visualization-WordCloud-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Production%20Ready-purple?style=for-the-badge" />
</p>

# 📖 Project Overview
This project performs Sentiment Analysis and Emotion Detection using text collected from three different real-world sources:
- Amazon Product Reviews
- Twitter (Airline Sentiment Dataset)
- News Headlines Dataset
The goal is to classify text into Positive, Neutral, or Negative, detect emotional tone, and compare public opinion patterns across multiple platforms.
This project was developed as part of the CodeAlpha Data Analytics Internship (Task-4).

# 🎯 Objectives :-
- Clean & preprocess text from 3 different datasets
- Perform rule-based sentiment analysis (VADER + TextBlob)
- Extract emotional categories using NRC Emotion Lexicon
- Convert Amazon review ratings → sentiment labels
- Train a TF-IDF + Logistic Regression sentiment classifier
- Evaluate model performance
- Compare sentiment distribution across Amazon, Twitter, and News text

<h2>📂 Project Structure
<pre> Consumer-Sentiment-Emotion-Analysis/
│
├── data/
│   ├── Amazon_Reviews.csv
│   ├── Tweets.csv
│   └── News_Category_Dataset_v3.json
│
├── models/
│   ├── logistic_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   └── Consumer_sentiment_analysis.ipynb
│
├── requirements.txt
└── README.md
</pre>

# 🧰 Tech Stack :- 
Programming Language: **Python**
Libraries used :
- Pandas
- NumPy
- NLTK
- VADER Sentiment
- TextBlob
- Scikit-Learn
- Matplotlib
- Seaborn
- WordCloud (optional)

# 🧹 Data Preprocessing Steps
- ✔ Lowercasing
- ✔ Removing URLs, mentions, hashtags
- ✔ Cleaning punctuation & special symbols
- ✔ Removing stopwords
- ✔ Lemmatization
- ✔ Combining multiple datasets into a unified dataframe

# 🧪 Sentiment Analysis Methods
1️⃣ Rule-Based Sentiment:
- VADER Sentiment Analyzer
- TextBlob polarity scoring

2️⃣ Machine Learning Sentiment Classifier
Label Mapping:
⭐ 4–5 → Positive
⭐ 3 → Neutral
⭐ 1–2 → Negative
  
Train ML model using:
- TF-IDF Vectorizer
- Logistic Regression
- Evaluation metrics used:
- Accuracy
- Precision, Recall, F1-Score
All trained models (TF-IDF + Logistic Regression) are exported using pickle for deployment.

# 📊 Visualizations Included :-
- Sentiment distribution across platforms
- Emotion count comparison
- WordCloud (positive & negative text)
- Confusion matrix of the ML model
- Clean bar charts for sentiment trend comparison


#👨‍💻 Developed By
<h2> Ayush 
<pre>
- 💼LinkedIn: https://linkedin.com/in/ayush130
- 💻GitHub: https://github.com/ayush13-0
- ✉️Email- bhanuseenu914@gmail.com

📜 License
- This project is licensed under the **MIT License**.
