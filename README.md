# Smart Spam Detection Using Machine Learning

## Project Overview
Smart Spam Detection is a machine learning project that classifies SMS messages into two categories: Spam and Ham (legitimate messages).

The system uses Natural Language Processing (NLP) and machine learning techniques to identify unwanted messages.

## Objectives
- Detect spam SMS messages automatically.
- Classify messages as Spam or Ham.
- Apply NLP techniques to text data.
- Build a simple interactive web application.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib

## Machine Learning
- TF-IDF Vectorization
- Naive Bayes / Trained classification model

## Project Structure
```text
SmartSpamDetection/
├── dataset/
├── app.py
├── main.py
├── spam_model.pkl
├── tfidf_vectorizer.pkl
├── model_metrics.pkl
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/sainallamala136-cmd/SmartSpamDetection.git
```

Install the required libraries:

```bash
pip install pandas numpy scikit-learn streamlit matplotlib
```

## Run the Application

```bash
streamlit run app.py
```

## Project Status
Development in progress.

## Author
Sai Ganesh

## License
To be decided.
## Model Performance

The Smart Spam Detection model was evaluated using the following metrics:

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 95.26% |
| Precision | 95.51% |
| Recall | 95.26% |
| F1-score | 94.78% |

### Confusion Matrix

| Actual / Predicted | Ham | Spam |
|---|---:|---:|
| Ham | 903 | 0 |
| Spam | 49 | 82 |

The model correctly classified 903 legitimate messages and 82 spam messages in the evaluation dataset. It missed 49 spam messages.

## Confusion Matrix Visualization

![Confusion Matrix](confusion_matrix.png)

## Streamlit Dashboard

![Smart Spam Detection Dashboard](dashboard.png)