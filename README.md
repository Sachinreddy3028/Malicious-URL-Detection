🔐 Malicious URL Detection using Machine Learning and LSTM
This project presents a hybrid machine learning and deep learning framework for real-time malicious URL detection. It integrates traditional ML classifiers (Random Forest, Logistic Regression, XGBoost, SVM) with a lightweight LSTM model to classify URLs as phishing, malware, defacement, or benign using lexical and sentiment-based features.

📌 Project Highlights
Real-time URL classification capability using a trained Random Forest model.

Feature engineering based on lexical analysis (e.g., URL length, dot count, HTTPS presence) and sentiment analysis.

Comparative evaluation of multiple ML models and a character-level LSTM model.

Visualization of model performance and feature distributions.

Lightweight LSTM network trained for rapid prediction on character-tokenized URL data.

📈 Visualizations
The repo includes:

Model performance comparison plots

Distribution of URL features (length, dot count, sentiment, HTTPS)

LSTM training accuracy/loss curves

🧠 Future Enhancements
Integrate WHOIS, DNS, and IP-based metadata

Explore Transformer-based models (e.g., BERT, URLTran)

Improve interpretability using SHAP/LIME

Deploy using federated learning or real-time streaming (Kafka/Spark)

🧑‍💻 Author
Sachin Radder
MSc in Cybersecurity
National College of Ireland
