import os
import json
import joblib
import torch
import numpy as np
import scipy.sparse
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# --- COLORS FOR TERMINAL OUTPUT ---
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

class ChampionTester:
    def __init__(self):
        self.model_path = 'saved_models'
        self.config_path = os.path.join(self.model_path, 'model_config.json')
        self.model_type = None
        self.model = None
        self.tokenizer = None 
        self.vectorizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.load_champion()

    def load_champion(self):
        print(f"{Colors.CYAN}🔄 Initializing System...{Colors.RESET}")
        
        if not os.path.exists(self.config_path):
            print(f"{Colors.RED}❌ Error: Configuration file not found.{Colors.RESET}")
            print("   Please run 'train.py' first to generate a champion model.")
            exit()

        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                self.model_type = config['model_type']
                name = config['name']
                acc = config['accuracy'] * 100
                print(f"{Colors.GREEN}✅ Champion Loaded: {Colors.BOLD}{name}{Colors.RESET} (Training Accuracy: {acc:.2f}%)")

            if self.model_type == 'bert':
                bert_path = os.path.join(self.model_path, 'champion_bert')
                self.tokenizer = DistilBertTokenizer.from_pretrained(bert_path)
                self.model = DistilBertForSequenceClassification.from_pretrained(bert_path).to(self.device)
                self.model.eval()
                print(f"   Engine: Deep Learning (Transformers) on {self.device}")
                
            elif self.model_type == 'sklearn':
                self.model = joblib.load(os.path.join(self.model_path, 'champion_model.pkl'))
                self.vectorizer = joblib.load(os.path.join(self.model_path, 'champion_vectorizer.pkl'))
                print("   Engine: Traditional ML (Scikit-Learn)")
                
        except Exception as e:
            print(f"{Colors.RED}❌ Critical Error loading model: {e}{Colors.RESET}")
            exit()

    def predict(self, text, rating=3, is_verified=0):
        if not self.model: return "Error", 0

        try:
            # --- BERT PREDICTION (Text Only) ---
            if self.model_type == 'bert':
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = F.softmax(outputs.logits, dim=1)
                    confidence, predicted_class = torch.max(probs, dim=1)
                
                label = "Real Review" if predicted_class.item() == 1 else "Fake Review"
                score = round(confidence.item() * 100, 2)
                return label, score

            # --- SKLEARN PREDICTION (Text + Metadata) ---
            elif self.model_type == 'sklearn':
                # 1. Text Features
                tfidf_features = self.vectorizer.transform([text])
                
                # 2. Metadata Features
                review_len = len(text.split())
                meta_features = np.array([[int(is_verified), int(rating), review_len]])
                
                # 3. Combine
                final_features = scipy.sparse.hstack([tfidf_features, meta_features])
                
                # 4. Predict
                prediction = self.model.predict(final_features)[0]
                label = "Real Review" if prediction == 1 else "Fake Review"
                
                # Try to get probability if supported
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(final_features)
                    score = round(np.max(probs) * 100, 2)
                else:
                    score = 99.9 
                    
                return label, score

        except Exception as e:
            print(f"{Colors.RED}Prediction Error: {e}{Colors.RESET}")
            return "Error", 0

# --- AUTOMATED TEST SUITE ---
def run_automated_tests():
    tester = ChampionTester()
    
    # Define Test Cases (10 Fake, 10 Real)
    test_cases = [
        # --- FAKE REVIEWS (Expected: Fake Review) ---
        {"text": "Amazing product! This is the best thing I have ever bought in my entire life. I highly recommend it to everyone. Five stars!!!", "rating": 5, "verified": 0, "expected": "Fake Review", "desc": "Generic Hype"},
        {"text": "Item very good quality fast shipping. Seller is nice person. I buy again for sure. Excellent gadget for the price.", "rating": 5, "verified": 0, "expected": "Fake Review", "desc": "Broken English Bot"},
        {"text": "Do not buy this garbage. It broke after one day. Go buy the Brand-X version instead, it is much better and cheaper.", "rating": 1, "verified": 0, "expected": "Fake Review", "desc": "Competitor Bash"},
        {"text": "I received this product for free and I honest say it is perfect. Great gift for christmas. Buy it now you will not regret!", "rating": 5, "verified": 0, "expected": "Fake Review", "desc": "Promotional Push"},
        {"text": "INCREDIBLE PERFORMANCE. WOW. I AM BLOWN AWAY BY THE QUALITY. ABSOLUTELY FANTASTIC PURCHASE.", "rating": 5, "verified": 0, "expected": "Fake Review", "desc": "Caps Lock Warrior"},
        {"text": "Good item.", "rating": 5, "verified": 0, "expected": "Fake Review", "desc": "Vague One-Liner"},
        {"text": "Best phone case phone cover protection cheap iphone case samsung case durable quality best seller fast delivery.", "rating": 5, "verified": 0, "expected": "Fake Review", "desc": "Keyword Stuffer"},
        {"text": "Limited time offer quality! I ordered three for my family and they all love it. Don't wait, get yours today!", "rating": 5, "verified": 0, "expected": "Fake Review", "desc": "Urgency Trap"},
        {"text": "The product arrived on time and as described. The packaging was good. I am satisfied with the purchase.", "rating": 4, "verified": 0, "expected": "Fake Review", "desc": "Copy-Paste Template"},
        {"text": "My life was a mess until I bought this blender. Now I have energy, my skin is clear, and I got a promotion at work. Miraculous!", "rating": 5, "verified": 0, "expected": "Fake Review", "desc": "Suspicious Story"},

        # --- REAL REVIEWS (Expected: Real Review) ---
        {"text": "The battery life is solid, lasting about 8 hours on a single charge. However, the screen brightness is a bit low when using it outdoors.", "rating": 4, "verified": 1, "expected": "Real Review", "desc": "Balanced Tech Review"},
        {"text": "I liked the design, but the zipper got stuck on the third day of use. I had to return it for a replacement.", "rating": 2, "verified": 1, "expected": "Real Review", "desc": "Specific Defect"},
        {"text": "Compared to the Sony WH-1000XM4, the noise cancellation isn't quite as strong, but for half the price, the sound quality is surprisingly clear.", "rating": 4, "verified": 1, "expected": "Real Review", "desc": "Comparison"},
        {"text": "I use this for my morning runs. It stays in my ears well, but the bluetooth connection sometimes drops if I put my phone in my back pocket.", "rating": 3, "verified": 1, "expected": "Real Review", "desc": "Usage Scenario"},
        {"text": "The fabric feels heavier than it looks in the picture. It's more of a winter coat than a light jacket, which is fine, but not what I expected.", "rating": 3, "verified": 1, "expected": "Real Review", "desc": "Material Description"},
        {"text": "Arrived a day late due to the snowstorm, but the box was undamaged. The setup instructions were a bit confusing though.", "rating": 4, "verified": 1, "expected": "Real Review", "desc": "Shipping Context"},
        {"text": "It does what it says. Not life-changing, but it cleans my hardwood floors effectively. Ideally, the cord would be a few feet longer.", "rating": 4, "verified": 1, "expected": "Real Review", "desc": "Verified Tone"},
        {"text": "Great concept, poor execution. The app is buggy and crashes often on iOS 15. Hardware is fine, but software needs an update.", "rating": 2, "verified": 1, "expected": "Real Review", "desc": "Constructive Criticism"},
        {"text": "Bought this for my 6-year-old. She loves the colors, but the plastic pieces feel a bit flimsy. Hopefully it lasts a few months.", "rating": 3, "verified": 1, "expected": "Real Review", "desc": "Gift Context"},
        {"text": "Update after 6 months: It's still working, but the non-stick coating is starting to chip on the edges. Still a decent pan for eggs.", "rating": 3, "verified": 1, "expected": "Real Review", "desc": "Long-Term Update"}
    ]

    print("\n" + "="*80)
    print(f"{Colors.BOLD}   🔍 AUTOMATED TEST SUITE RUNNING...{Colors.RESET}")
    print("="*80)
    print(f"{'TYPE':<10} | {'TEST CASE DESCRIPTION':<25} | {'PREDICTION':<15} | {'CONF':<6} | {'RESULT'}")
    print("-" * 80)

    passed = 0
    
    for i, case in enumerate(test_cases):
        # Run Predict
        label, conf = tester.predict(case['text'], case['rating'], case['verified'])
        
        # Check correctness
        is_correct = (label == case['expected'])
        if is_correct:
            passed += 1
            status_icon = f"{Colors.GREEN}✔ PASS{Colors.RESET}"
        else:
            status_icon = f"{Colors.RED}✘ FAIL{Colors.RESET}"
            
        # Coloring the label output
        label_display = f"{Colors.GREEN}{label}{Colors.RESET}" if label == "Real Review" else f"{Colors.RED}{label}{Colors.RESET}"
        expected_type = "FAKE" if case['expected'] == "Fake Review" else "REAL"
        
        print(f"{expected_type:<10} | {case['desc']:<25} | {label_display:<24} | {int(conf)}%   | {status_icon}")

    # Summary
    score = (passed / len(test_cases)) * 100
    color = Colors.GREEN if score >= 80 else (Colors.YELLOW if score >= 60 else Colors.RED)
    
    print("-" * 80)
    print(f"📊 TEST COMPLETE: {passed}/{len(test_cases)} Passed")
    print(f"🏆 FINAL ACCURACY: {color}{Colors.BOLD}{score:.1f}%{Colors.RESET}")
    print("=" * 80)

if __name__ == "__main__":
    run_automated_tests()