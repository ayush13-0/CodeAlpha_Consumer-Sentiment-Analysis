# 🔍Consumer Sentiment & Emotion Analysis of Multi-Source Data (Amazon + Twitter + News)  
📜 CodeAlpha Internship — Task 4

# 📖 Overview
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
- Ayush
📧 Email: bhanuseenu914@gmail.com
- 🔗 LinkedIn: https://linkedin.com/in/ayush130
- 🔗 GitHub: https://github.com/ayush13-0

📜 License
- This project is licensed under the **MIT License**.
