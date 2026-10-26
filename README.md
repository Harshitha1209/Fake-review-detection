# Fake Product Review Detection Using Machine Learning and Deep Learning

## Project Overview

Fake Product Review Detection is a web-based application designed to identify whether a product review is **Real or Fake** using Machine Learning and Deep Learning techniques. The application analyzes user-provided reviews and displays a prediction with a confidence score.

The system evaluates multiple classification models and selects a **Champion Model** based on the chosen evaluation criteria. This model selection approach helps identify a suitable model for making predictions on submitted product reviews.

The project aims to help users identify potentially misleading product reviews and make more informed online purchasing decisions.

## Objectives

- Detect potentially fake and genuine product reviews.
- Compare multiple Machine Learning and Deep Learning models.
- Select the Champion Model based on evaluation results and defined selection criteria.
- Build an interactive web application using Flask.
- Display prediction results and confidence scores.
- Provide a simple interface for users to submit and analyze reviews.

## Technologies Used

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Deep Learning:** PyTorch, Hugging Face Transformers
- **Natural Language Processing:** TF-IDF and DistilBERT
- **Web Framework:** Flask
- **Data Processing:** NumPy, SciPy
- **Database:** SQLite
- **Frontend:** HTML, CSS
- **Tools:** VS Code, Git, GitHub

## Machine Learning and Deep Learning Models

The project explores the following models for fake review classification:

- Logistic Regression
- Random Forest
- Linear Support Vector Machine (Linear SVM)
- Gradient Boosting
- Naive Bayes
- DistilBERT

Traditional Machine Learning models use TF-IDF text features and additional review-related features, such as rating, verified purchase status, and review length, where applicable.

DistilBERT is used to explore contextual representations of review text for classification.

### Champion Model Selection

The application uses a Champion Model selection approach to identify the most suitable model from the evaluated candidates.

The selection process involves:

1. Training candidate classification models.
2. Evaluating their performance using appropriate classification metrics.
3. Comparing evaluation results against the defined selection criteria.
4. Selecting the best-performing eligible model as the Champion Model.
5. Using the selected model for review predictions in the application.

Evaluation may include accuracy, precision, recall, F1-score, and the confusion matrix, depending on the implemented evaluation workflow.

The Champion Model is intended to provide a consistent model selection and prediction process. Its actual performance depends on the dataset, evaluation methodology, and selection criteria.

## Key Features

- **Fake Review Classification:** Predicts whether a submitted review is classified as real or fake.
- **Champion Model Selection:** Selects a suitable model based on evaluation results and configured criteria.
- **Confidence Score:** Displays a prediction confidence score where supported by the implementation.
- **Web Interface:** Provides an interactive interface for submitting and analyzing reviews.
- **User Authentication:** Supports user registration and login functionality.
- **Model Training and Testing:** Provides separate scripts for model training and testing.
- **Database Integration:** Uses SQLite for application data.
- **Modular Design:** Separates the Flask application, model training, testing, and HTML templates.

## Project Structure

```text
FakeReviewDetector/
│
├── app.py
├── train.py
├── test.py
├── req.txt
├── .gitignore
├── README.md
│
├── templates/
│   ├── admin.html
│   ├── base.html
│   ├── base_paper.html
│   ├── index.html
│   ├── info.html
│   ├── login.html
│   ├── predict.html
│   ├── register.html
│   └── train.html
│
├── amazon_reviews.txt
├── test.txt
├── results/
├── saved_models/
└── instance/
```

**Note:** The dataset, local database, generated results, and trained model files may be excluded from the public repository depending on their size, privacy, and licensing requirements. Empty folders are not tracked by Git.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Harshitha1209/Fake-review-detection.git
cd Fake-review-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r req.txt
```

### 4. Prepare the Dataset and Models

Ensure that the required dataset is available at the path expected by the training script.

Review `train.py` to understand the training workflow and Champion Model selection criteria. Generate or obtain the required trained model artifacts before running the application if they are not included in the repository.

### 5. Run the Application

```bash
python app.py
```

Open the local URL displayed in the terminal. If Flask uses its default configuration, the address is usually:

```text
http://127.0.0.1:5000/
```

## How It Works

1. The user submits a product review through the web interface.
2. The application preprocesses the review text and applicable features.
3. Candidate models are trained and evaluated through the model training workflow.
4. The Champion Model is selected according to the configured evaluation criteria.
5. The selected model analyzes the submitted review.
6. The application displays the predicted class and, where implemented, a confidence score.

## Applications

- E-commerce review analysis
- Identification of potentially misleading customer feedback
- Online shopping decision support
- Natural Language Processing research
- Comparative evaluation of Machine Learning and Deep Learning models

## Limitations

- Predictions depend on the quality and representativeness of the training data.
- Performance may vary across products, platforms, and review styles.
- A confidence score does not guarantee that a prediction is correct.
- The Champion Model's effectiveness depends on the evaluation metrics and selection criteria.
- Predictions should not be treated as definitive proof that a review is fraudulent.

## Future Enhancements

- Improve classification performance using larger and more diverse datasets.
- Explore additional Transformer-based language models.
- Improve Champion Model selection using cross-validation and systematic model comparison.
- Develop an API for integration with other applications.
- Evaluate the system on reviews from different e-commerce platforms.
- Improve model interpretability and explainability.
- Deploy the application to a cloud platform.

## Author

**Harshitha Vallepu**

GitHub: [Harshitha1209](https://github.com/Harshitha1209)

Project Repository: [Fake Review Detection](https://github.com/Harshitha1209/Fake-review-detection)

## Disclaimer

This project is intended for educational and research purposes. Predictions are model-generated estimates and should be used as decision-support information rather than definitive proof that a product review is genuine or fake.
