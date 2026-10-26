from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
import time
import os
import json
import joblib
import numpy as np
import scipy.sparse
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# --- CONFIGURATION ---
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# --- DATABASE MODEL ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- CHAMPION PREDICTOR ---
class ChampionPredictor:
    def __init__(self):
        self.model_type = None
        self.model = None
        self.tokenizer = None 
        self.vectorizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.load_champion()

    def load_champion(self):
        config_path = os.path.join(MODEL_DIR, 'model_config.json')
        os.makedirs(MODEL_DIR, exist_ok=True)

        if not os.path.exists(config_path):
            app.logger.error("No trained model found at %s. Run train.py first.", config_path)
            self.model = None
            return

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.model_type = config.get('model_type', 'sklearn')
                app.logger.info("Loading champion model: %s", self.model_type)

            if self.model_type == 'bert':
                bert_dir = os.path.join(MODEL_DIR, 'champion_bert')
                self.tokenizer = DistilBertTokenizer.from_pretrained(bert_dir)
                self.model = DistilBertForSequenceClassification.from_pretrained(bert_dir).to(self.device)
                self.model.eval()
            elif self.model_type == 'sklearn':
                self.model = joblib.load(os.path.join(MODEL_DIR, 'champion_model.pkl'))
                self.vectorizer = joblib.load(os.path.join(MODEL_DIR, 'champion_vectorizer.pkl'))
            else:
                raise ValueError(f"Unsupported model type in {config_path}: {self.model_type}")
        except Exception as e:
            app.logger.exception("Failed to load the champion model: %s", e)
            self.model = None

    def predict(
        self,
        text: str,
        rating: int | None = None,
        is_verified: bool | None = None
    ) -> tuple[str, float]:
        if not text or not text.strip():
            raise ValueError("Review text cannot be empty.")
        if self.model is None:
            raise RuntimeError("No trained model is loaded. Run train.py before analyzing reviews.")

        if self.model_type == 'bert':
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, padding=True, max_length=128
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=1)
                confidence, predicted_class = torch.max(probabilities, dim=1)

            label = "Real Review" if predicted_class.item() == 1 else "Fake Review"
            return label, round(confidence.item() * 100, 2)

        if self.model_type == 'sklearn':
            if self.vectorizer is None:
                raise RuntimeError("The trained text vectorizer is not loaded.")
            if rating is not None and not 1 <= int(rating) <= 5:
                raise ValueError("Rating must be between 1 and 5.")

            tfidf_features = self.vectorizer.transform([text])
            review_len = len(text.split())
            verification_values = (0, 1) if is_verified is None else (int(is_verified),)
            class_probabilities = []

            rating_values = range(1, 6) if rating is None else (int(rating),)
            for rating_value in rating_values:
                for verified_value in verification_values:
                    meta_features = np.array([[verified_value, rating_value, review_len]])
                    features = scipy.sparse.hstack([tfidf_features, meta_features])
                    if hasattr(self.model, 'predict_proba'):
                        class_probabilities.append(self.model.predict_proba(features)[0])
                    else:
                        prediction = self.model.predict(features)[0]
                        probabilities = np.zeros(len(self.model.classes_))
                        class_index = np.flatnonzero(self.model.classes_ == prediction)[0]
                        probabilities[class_index] = 1.0
                        class_probabilities.append(probabilities)

            probabilities = np.mean(class_probabilities, axis=0)
            prediction = self.model.classes_[np.argmax(probabilities)]
            score = float(round(max(probabilities) * 100, 2)) if hasattr(self.model, 'predict_proba') else 75.0
            label = "Real Review" if prediction == 1 else "Fake Review"
            return label, score

        raise RuntimeError(f"Unsupported loaded model type: {self.model_type}")

# Initialize
predictor = ChampionPredictor()

# --- ROUTES ---
@app.route('/')
def home(): return render_template('index.html')

@app.route('/base_paper')
def base_paper(): return render_template('base_paper.html')

@app.route('/info')
def info(): return render_template('info.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        flash("Already logged in!", "info")
        return redirect(url_for('login'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        role = request.form.get('role')
        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        
        if User.query.filter_by(username=username).first():
            flash('Username exists.', 'danger')
        else:
            user = User(username=username, password=hashed_pw, is_admin=(role == 'admin'))
            db.session.add(user)
            db.session.commit()
            flash('Account created!', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('home'))
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and bcrypt.check_password_hash(user.password, request.form.get('password')):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Login failed.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/admin')
@login_required
def admin_dashboard():
    if not current_user.is_admin: return redirect(url_for('home'))
    return render_template('admin.html', users=User.query.all())

@app.route('/admin/delete/<int:user_id>')
@login_required
def delete_user(user_id):
    if not current_user.is_admin: return redirect(url_for('home'))
    user = User.query.get(user_id)
    if user and user.id != current_user.id:
        db.session.delete(user)
        db.session.commit()
        flash('User deleted.', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    label, score, text = None, 0, ""
    rating = 3
    is_verified = False
    
    if request.method == 'POST':
        text = request.form.get('review_text', '')
        try:
            rating_value = request.form.get('rating', 'unknown')
            rating = None if rating_value == 'unknown' else int(rating_value)
            verification = request.form.get('is_verified', 'unknown')
            if verification not in {'unknown', 'yes', 'no'}:
                raise ValueError("Invalid purchase verification value.")
            is_verified = {'unknown': None, 'yes': True, 'no': False}[verification]

            label, score = predictor.predict(text, rating, is_verified)
        except Exception:
            app.logger.exception("Review prediction failed")
            flash("Could not analyze this review. Check the rating and trained model, then try again.", "danger")
        
    return render_template('predict.html', prediction=label, confidence=score, review_text=text)

@app.route('/training')
@login_required
def training_page(): return render_template('train.html')

@app.route('/api/train', methods=['POST'])
@login_required
def run_dummy_training():
    time.sleep(3)
    return jsonify({"status": "success", "message": "Champion Model Retrained!"})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='admin').first():
            admin = User(username='admin', password=bcrypt.generate_password_hash('admin123').decode('utf-8'), is_admin=True)
            db.session.add(admin)
            db.session.commit()
    app.run(debug=True)