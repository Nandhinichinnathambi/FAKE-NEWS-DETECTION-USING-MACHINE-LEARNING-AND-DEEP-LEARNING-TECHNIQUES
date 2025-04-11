 📰 FAKE-NEWS-DETECTION-USING-MACHINE-LEARNING-AND-DEEP-LEARNING-TECHNIQUES
  This project focuses on detecting whether a news article is real or fake using a hybrid approach that combines traditional machine learning and advanced deep learning models. It enhances media authenticity by automatically classifying news content and includes real-time manual testing, ensemble learning, and insightful visualizations.

📌 Key Features
Uses multiple models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LSTM, and BERT

Applies advanced text preprocessing: stopword removal, lemmatization, TF-IDF, Word2Vec, and BERT embeddings

Supports real-time manual testing of news articles

Implements dimensionality reduction and 3D visualizations using PCA and t-SNE

Includes ensemble methods (Voting Classifier) and hyperparameter tuning

Evaluates models using Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix

🛠 Technologies Used
Python 3.x

Libraries: Scikit-learn, TensorFlow, Keras, NLTK, Transformers, Matplotlib, Seaborn, Plotly

Development Environment: Google Colab

📂 Dataset Description
Collected from Malai Murasu, Tamil Murasu, and other verified sources (2024–2025)

Stored in CSV format with columns: title, text, date, label

Labels: REAL or FAKE

Preprocessed with text cleaning, vectorization, and SMOTE for balancing

🔄 Methodology Summary
Data Collection and Preprocessing

Feature Extraction using TF-IDF, Word2Vec, and BERT

Dimensionality Reduction using PCA and t-SNE

Model Training using machine learning and deep learning models

Evaluation with classification metrics and ROC curves

Real-time manual news testing interface with model comparison

Deployment-ready architecture using Flask or Streamlit

✅ Results Snapshot
Machine learning models achieved strong performance, with Logistic Regression scoring over 96% accuracy

Deep learning models, especially BERT, achieved the highest accuracy at 98.9%

Ensemble techniques further boosted prediction consistency and robustness

🧪 Real-Time Testing
Users can input their own news content into the system and get instant predictions from multiple trained models. Results are visualized in an easy-to-understand bar chart format.

🚀 Scalability & Deployment
Designed for web deployment using Flask, Streamlit, or FastAPI

Can be scaled using cloud platforms like AWS or Google Cloud

Supports multilingual and large-scale dataset expansion

👩‍💻 Author
Nandhini.C

