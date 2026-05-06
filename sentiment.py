# sentiment.py - OPTIMIZED VERSION 6.1 (WITH TIME SERIES)
"""
Analisis Sentimen Multi-Model Ensemble V6.1
Dengan:
- Balanced bias (tidak terlalu positif/negatif)
- Mixed sentiment presisi
- Confidence calibration
- Hard rules untuk kasus umum
- Auto retrain RF
- TIME SERIES ANALYSIS untuk grafik harian
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import torch.nn.functional as F
import re
import numpy as np
import time
import pickle
import os
import json
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ===============================
# KONFIGURASI MODEL
# ===============================
MODEL_NAME = "indobenchmark/indobert-base-p1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
MAX_LENGTH = 128

SIMPLE_MODEL_NAME = "w11wo/indonesian-roberta-base-sentiment-classifier"

# 🔥 BALANCED BIAS
BERT_WEIGHT = 0.50
RULE_WEIGHT = 0.35
RF_WEIGHT = 0.15

# 🔥 OPTIMIZED PARAMETERS
NEGATIVE_MULTIPLIER = 0.7
POSITIVE_BOOST = 1.2
POSITIVE_BIAS = 0.2  # 🔧 FIX: diturunkan dari 0.8 agar tidak terlalu bias ke Positif

# Path untuk menyimpan model
RF_MODEL_PATH = "models/rf_sentiment_validator.pkl"
TRAINING_DATA_PATH = "models/training_data.jsonl"
ERROR_LOG_PATH = "models/error_log.txt"
EVAL_RESULTS_PATH = "models/evaluation_results.json"

# ===============================
# SAFE PRINT FOR WINDOWS
# ===============================
def safe_print(*args, **kwargs):
    try:
        import sys
        print(*args, **kwargs)
        sys.stdout.flush()
    except UnicodeEncodeError:
        clean_args = [str(arg).encode('ascii', 'ignore').decode('ascii') for arg in args]
        print(*clean_args, **kwargs)

safe_print("="*60)
safe_print("LOADING OPTIMIZED SENTIMENT SYSTEM (V6.1 - TIME SERIES)")
safe_print("="*60)
safe_print(f"Advanced Model: {MODEL_NAME}")
safe_print(f"Simple Model: {SIMPLE_MODEL_NAME}")
safe_print(f"Device: {DEVICE}")
safe_print(f"Negative Multiplier: {NEGATIVE_MULTIPLIER}")
safe_print(f"Positive Boost: {POSITIVE_BOOST}")
safe_print(f"Positive Bias: {POSITIVE_BIAS}")

# Load Advanced Model
try:
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        return_dict=True,
        num_labels=3
    )
    model.eval()
    model = model.to(DEVICE)
    load_time = time.time() - start_time
    safe_print(f"[OK] Advanced model loaded in {load_time:.2f}s")
except Exception as e:
    safe_print(f"[WARN] Error loading IndoBERT: {e}")
    MODEL_NAME = "w11wo/indonesian-roberta-base-sentiment-classifier"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    model = model.to(DEVICE)
    safe_print(f"[OK] Using fallback advanced model: {MODEL_NAME}")

# Load Simple Model
try:
    simple_pipeline = pipeline(
        "sentiment-analysis",
        model=SIMPLE_MODEL_NAME,
        tokenizer=SIMPLE_MODEL_NAME,
        device=0 if torch.cuda.is_available() else -1
    )
    safe_print(f"[OK] Simple model loaded: {SIMPLE_MODEL_NAME}")
except Exception as e:
    safe_print(f"[WARN] Error loading simple model: {e}")
    simple_pipeline = None

LABELS = ["Negatif", "Netral", "Positif"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABELS)}

# ===============================
# TIME SERIES DATA STORAGE (🔥 BARU)
# ===============================
# Untuk menyimpan data sentimen dengan timestamp
sentiment_history = []  # List of dict dengan field: date, sentiment, text

def add_to_history(text, sentiment, confidence):
    """Tambahkan hasil analisis ke history dengan timestamp"""
    global sentiment_history
    sentiment_history.append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "datetime": datetime.now().isoformat(),
        "text": text[:100],  # Simpan 100 karakter pertama
        "sentiment": sentiment,
        "confidence": confidence
    })
    
    # Batasi history (simpan 10000 terakhir)
    if len(sentiment_history) > 10000:
        sentiment_history = sentiment_history[-10000:]

def get_daily_sentiment(days=30):
    """
    Agregasi sentimen per hari
    
    Args:
        days: Jumlah hari ke belakang
    
    Returns:
        Dictionary dengan tanggal sebagai key dan statistik sentimen
    """
    global sentiment_history
    
    # Inisialisasi result
    result = {}
    
    # Buat date range untuk days terakhir
    today = datetime.now().date()
    for i in range(days):
        date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        result[date] = {
            "Positif": 0,
            "Negatif": 0,
            "Netral": 0,
            "total": 0
        }
    
    # Agregasi dari history
    for item in sentiment_history:
        date = item["date"]
        sentiment = item["sentiment"]
        if date in result:
            result[date][sentiment] += 1
            result[date]["total"] += 1
    
    return result

def format_trend_chart(days=30):
    """
    Format data untuk chart line
    
    Returns:
        Dictionary dengan labels, positif, negatif, netral
    """
    daily_data = get_daily_sentiment(days)
    
    # Urutkan berdasarkan tanggal (dari yang paling lama ke terbaru)
    dates = sorted(daily_data.keys())
    
    # 🔧 FIX: urutan ascending (kiri=lama, kanan=baru) — tidak dibalik
    return {
        "labels": dates,
        "positif": [daily_data[d]["Positif"] for d in dates],
        "negatif": [daily_data[d]["Negatif"] for d in dates],
        "netral": [daily_data[d]["Netral"] for d in dates],
        "total": [daily_data[d]["total"] for d in dates]
    }

def clear_history():
    """Clear sentiment history (untuk testing)"""
    global sentiment_history
    sentiment_history = []
    safe_print("Sentiment history cleared")

def get_history_stats():
    """Dapatkan statistik dari history"""
    global sentiment_history
    if not sentiment_history:
        return {
            "total": 0,
            "positif": 0,
            "negatif": 0,
            "netral": 0,
            "date_range": None
        }
    
    dates = [item["date"] for item in sentiment_history]
    return {
        "total": len(sentiment_history),
        "positif": sum(1 for item in sentiment_history if item["sentiment"] == "Positif"),
        "negatif": sum(1 for item in sentiment_history if item["sentiment"] == "Negatif"),
        "netral": sum(1 for item in sentiment_history if item["sentiment"] == "Netral"),
        "date_range": {
            "start": min(dates),
            "end": max(dates)
        }
    }

# ===============================
# COMMON PATTERNS (HARD RULES)
# ===============================
COMMON_PATTERNS = {
    "biasa aja": "Netral",
    "biasa saja": "Netral",
    "lumayan": "Netral",
    "standar": "Netral",
    "not bad": "Positif",
    "ga jelek": "Positif",
    "tidak jelek": "Positif",
    "ga buruk": "Positif",
    "tidak buruk": "Positif",
    "ga bagus": "Negatif",
    "tidak bagus": "Negatif",
    "ga keren": "Negatif",
    "tidak keren": "Negatif",
    "oke lah": "Netral",
    "ok lah": "Netral",
    "b aja": "Netral",
}

# ===============================
# EMOJI DATABASE
# ===============================
POSITIVE_EMOJIS = ["😇", "👏", "👍", "❤️", "🔥", "😍", "🥰", "😊", "😁", "😂", "🤣", "💪", "🎉", "✨", "⭐", "🏆"]
NEGATIVE_EMOJIS = ["😡", "🤮", "👎", "💀", "😤", "😠", "😭", "😢", "💔", "🤬", "😱", "😨", "👿", "💩", "🤢", "😖"]

# ===============================
# FAKE NEGATIVE DETECTION
# ===============================
FAKE_NEGATIVE_PATTERNS = [
    "parah keren", "parah bagus", "parah mantap",
    "gila bagus", "gila keren", "gila mantap",
    "gokil banget", "edan keren",
]

class RandomForestValidator:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self._load_model()
    
    def _load_model(self):
        if os.path.exists(RF_MODEL_PATH):
            try:
                with open(RF_MODEL_PATH, 'rb') as f:
                    saved = pickle.load(f)
                    self.model = saved['model']
                    self.scaler = saved['scaler']
                    self.is_trained = True
                safe_print(f"[OK] Random Forest validator loaded")
            except Exception as e:
                safe_print(f"[WARN] Could not load RF model: {e}")
    
    def save_model(self):
        os.makedirs(os.path.dirname(RF_MODEL_PATH), exist_ok=True)
        with open(RF_MODEL_PATH, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
        safe_print(f"[OK] Random Forest validator saved")
    
    def extract_features(self, text, bert_scores, rule_score, emoji_score):
        text_lower = text.lower()
        words = text_lower.split()
        
        features = [
            len(words), len(text),
            sum(1 for w in words if w in POSITIVE_KEYWORDS),
            sum(1 for w in words if w in NEGATIVE_KEYWORDS),
            sum(1 for w in words if w in NEGATION_WORDS),
            sum(1 for w in words if w in INTENSIFIERS),
            text.count('!'), text.count('?'),
            sum(1 for w in words if len(w) > 5),
            sum(1 for w in words if w.isupper()),
            rule_score, emoji_score,
            bert_scores.get("Positif", 0),
            bert_scores.get("Negatif", 0),
            bert_scores.get("Netral", 0),
        ]
        return features
    
    def predict(self, features):
        if not self.is_trained or self.model is None:
            return None, 0.0
        try:
            features_scaled = self.scaler.transform([features])
            pred = self.model.predict(features_scaled)[0]
            proba = self.model.predict_proba(features_scaled)[0]
            return pred, max(proba)
        except Exception as e:
            return None, 0.0
    
    def train(self, X_train, y_train):
        safe_print("\n[PROCESS] Training Random Forest Validator...")
        X_scaled = self.scaler.fit_transform(X_train)
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=15,
            min_samples_split=5, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
        train_acc = (self.model.predict(X_scaled) == y_train).mean()
        safe_print(f"[OK] Random Forest trained! Accuracy: {train_acc:.4f}")
        self.save_model()
        return train_acc

rf_validator = RandomForestValidator()

# ===============================
# KEYWORD DATABASE
# ===============================
POSITIVE_KEYWORDS = [
    "alhamdulillah", "masyaallah", "aamiin", "amin", "subhanallah",
    "barakallah", "insyaallah", "luar biasa", "mantap", "keren", "hebat",
    "sukses", "juara", "sehat", "bagus", "baik", "top", "jos", "respect",
    "salut", "terbaik", "amazing", "bangga", "support", "mantul",
    "setuju", "sepakat", "cakep", "ganteng", "cantik", "indah",
    "berkualitas", "recommended", "worth it", "gacor", "gokil",
    "senang", "bahagia", "suka", "cinta", "love", "gemes", "lucu",
    "ngakak", "wow", "sempurna", "puas", "terharu", "semangat"
]

NEGATIVE_KEYWORDS = [
    "jelek", "buruk", "bohong", "benci", "gagal", "sial", "tolol",
    "bodoh", "anjing", "kampret", "payah", "malas", "cacat", "goblok",
    "keparat", "bangsat", "setan", "bejat", "jahat", "kontol", "memek",
    "pantek", "jancok", "jancuk", "ngentot", "menipu", "bohongin",
    "hoax", "fitnah", "sampah", "rusak", "korupsi", "garing", "nyesel",
    "nyesek", "jengkel", "kesel", "kecewa", "frustasi", "stress",
    "males", "capek", "lelah", "marah", "geram", "muak", "jijik", "sedih"
]

NEGATION_WORDS = ["tidak", "tak", "bukan", "ga", "gak", "nggak", "enggak", "ndak", "kaga"]
INTENSIFIERS = ["banget", "bgt", "sekali", "sangat", "sgt", "super", "luar biasa"]

STRONG_POSITIVE = ["alhamdulillah", "masyaallah", "aamiin", "amin", "subhanallah"]
STRONG_NEGATIVE = ["anjing", "bangsat", "kontol", "memek", "pantek"]

# ===============================
# PREPROCESSING FUNCTIONS
# ===============================
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def emoji_score(text):
    if not isinstance(text, str):
        return 0
    score = 0
    for e in POSITIVE_EMOJIS:
        if e in text:
            score += 1
    for e in NEGATIVE_EMOJIS:
        if e in text:
            score -= 1
    return score

def apply_hard_rules(text):
    """Hard rules untuk kasus umum"""
    if not isinstance(text, str):
        return None, None
    
    text_lower = text.lower()
    for pattern, label in COMMON_PATTERNS.items():
        if pattern in text_lower:
            safe_print(f"   📋 Hard rule matched: '{pattern}' -> {label}")
            return label, 85.0
    
    return None, None

def has_mixed_sentiment(text, rule_score):
    """Deteksi mixed sentiment"""
    if not isinstance(text, str):
        return False, None
    
    text_lower = text.lower()
    contrast_words = ["tapi", "namun", "tetapi", "sedangkan", "sementara"]
    
    for word in contrast_words:
        if word in text_lower:
            parts = re.split(r'tapi|namun|tetapi|sedangkan|sementara', text_lower)
            if len(parts) >= 2:
                after = parts[1]
                if any(w in after for w in NEGATIVE_KEYWORDS):
                    return True, "Negatif"
                elif any(w in after for w in POSITIVE_KEYWORDS):
                    return True, "Positif"
    
    return False, None

def rule_based_score(text):
    if not isinstance(text, str):
        return 0, False
    
    text_lower = text.lower()
    score = 0
    has_keyword = False
    
    for word in POSITIVE_KEYWORDS:
        if word in text_lower:
            score += 1
            has_keyword = True
    
    for word in NEGATIVE_KEYWORDS:
        if word in text_lower:
            score -= 1 * NEGATIVE_MULTIPLIER
            has_keyword = True
    
    if score > 0:
        score += POSITIVE_BOOST
    
    score += POSITIVE_BIAS
    score += emoji_score(text)
    
    words = text_lower.split()
    for i, word in enumerate(words):
        if word in NEGATION_WORDS and i < len(words) - 1:
            next_word = words[i + 1]
            if next_word in POSITIVE_KEYWORDS:
                score -= 2
            elif next_word in NEGATIVE_KEYWORDS:
                score += 2
    
    for intens in INTENSIFIERS:
        if intens in text_lower:
            if score > 0:
                score += 1
            elif score < 0:
                score += 0.5
    
    confidence_boost = has_keyword and abs(score) >= 2
    return score, confidence_boost

def calibrate_confidence(label, confidence):
    """Kalibrasi confidence"""
    # 🔧 FIX: threshold Negatif diturunkan 70→57 agar tidak salah klasifikasi
    if label == "Negatif" and confidence < 57:
        return "Netral", 55
    if label == "Positif" and confidence < 60:
        return "Netral", 55
    if label == "Netral" and confidence < 50:
        return "Netral", 50
    return label, confidence

def smart_strong_rule_override(text):
    if not isinstance(text, str):
        return None, None
    
    text_lower = text.lower()
    word_count = len(text_lower.split())
    
    for word in STRONG_POSITIVE:
        if word in text_lower and word_count <= 5:
            return "Positif", 90.0
    
    for word in STRONG_NEGATIVE:
        if word in text_lower and word_count <= 3:
            return "Negatif", 90.0
    
    return None, None

def short_comment_analysis(text, rule_score):
    if not isinstance(text, str):
        return None, None
    
    words = text.split()
    word_count = len(words)
    
    if word_count <= 2:
        if rule_score > 0:
            return "Positif", 80.0
        elif rule_score < -1:
            return "Negatif", 80.0
        else:
            return "Netral", 70.0
    
    return None, None

def advanced_transformer_predict(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                          padding=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            
            confidence, predicted_class = torch.max(probs, dim=1)
            label = ID_TO_LABEL[predicted_class.item()]
            
            scores = {
                "Negatif": round(probs[0][0].item() * 100, 2),
                "Netral": round(probs[0][1].item() * 100, 2),
                "Positif": round(probs[0][2].item() * 100, 2)
            }
            return label, round(confidence.item() * 100, 2), scores
    except Exception as e:
        safe_print(f"[WARN] Advanced transformer error: {e}")
        return "Netral", 50.0, {"Negatif": 33.33, "Netral": 33.34, "Positif": 33.33}

def simple_sentiment(text):
    if not simple_pipeline:
        return "Netral", 50.0
    
    try:
        text_clean = normalize_text(text)
        if not text_clean:
            return "Netral", 50.0
        
        result = simple_pipeline(text_clean[:256])[0]
        label = result["label"].lower()
        score = result["score"] * 100
        
        if label == "positive":
            return "Positif", score
        elif label == "negative":
            if score < 70:
                return "Netral", score
            return "Negatif", score
        else:
            return "Netral", score
    except Exception as e:
        safe_print(f"[WARN] Simple model error: {e}")
        return "Netral", 50.0

def advanced_ensemble_predict(text, bert_label, bert_confidence, bert_scores, rule_score, emoji_val):
    weighted_scores = {
        "Positif": (bert_scores["Positif"] * BERT_WEIGHT) + 
                   (bert_confidence if bert_label == "Positif" else 0) * RULE_WEIGHT,
        "Negatif": (bert_scores["Negatif"] * BERT_WEIGHT) + 
                   (bert_confidence if bert_label == "Negatif" else 0) * RULE_WEIGHT,
        "Netral": (bert_scores["Netral"] * BERT_WEIGHT) + 
                  (bert_confidence if bert_label == "Netral" else 0) * RULE_WEIGHT
    }
    
    total = sum(weighted_scores.values())
    if total > 0:
        weighted_scores = {k: round(v / total * 100, 2) for k, v in weighted_scores.items()}
    
    final_label = max(weighted_scores, key=weighted_scores.get)
    final_confidence = weighted_scores[final_label]
    
    if abs(rule_score) >= 4:
        override_label = "Positif" if rule_score > 0 else "Negatif"
        if override_label != final_label:
            final_label = override_label
            final_confidence = 85.0
    
    if rf_validator.is_trained:
        try:
            features = rf_validator.extract_features(text, bert_scores, rule_score, emoji_val)
            rf_pred, rf_conf = rf_validator.predict(features)
            if rf_pred is not None and rf_conf > 0.65 and rf_pred != LABEL_TO_ID[final_label]:
                rf_label = ID_TO_LABEL[rf_pred]
                if rf_conf > 0.75:
                    adjusted_scores = {
                        "Positif": weighted_scores.get("Positif", 0) * (1 - RF_WEIGHT) + 
                                   (100 if rf_label == "Positif" else 0) * RF_WEIGHT,
                        "Negatif": weighted_scores.get("Negatif", 0) * (1 - RF_WEIGHT) + 
                                   (100 if rf_label == "Negatif" else 0) * RF_WEIGHT,
                        "Netral": weighted_scores.get("Netral", 0) * (1 - RF_WEIGHT) + 
                                  (100 if rf_label == "Netral" else 0) * RF_WEIGHT
                    }
                    total_adj = sum(adjusted_scores.values())
                    if total_adj > 0:
                        final_label = max(adjusted_scores, key=adjusted_scores.get)
                        final_confidence = round(adjusted_scores[final_label] / total_adj * 100, 2)
        except Exception as e:
            safe_print(f"[WARN] RF error: {e}")
    
    return final_label, final_confidence

def advanced_analyze(text, return_details=False):
    if not isinstance(text, str) or not text.strip():
        if return_details:
            return "Netral", 0.0, {"method": "fallback"}
        return "Netral", 0.0
    
    text = normalize_text(text)
    if not text:
        if return_details:
            return "Netral", 0.0, {"method": "fallback"}
        return "Netral", 0.0
    
    hard_label, hard_conf = apply_hard_rules(text)
    if hard_label:
        if return_details:
            return hard_label, hard_conf, {"method": "hard_rule"}
        return hard_label, hard_conf
    
    strong_label, strong_conf = smart_strong_rule_override(text)
    if strong_label:
        if return_details:
            return strong_label, strong_conf, {"method": "strong_rule"}
        return strong_label, strong_conf
    
    rule_score, rule_boost = rule_based_score(text)
    emoji_val = emoji_score(text)
    
    is_mixed, mixed_label = has_mixed_sentiment(text, rule_score)
    if is_mixed and mixed_label:
        if return_details:
            return mixed_label, 80.0, {"method": "mixed_sentiment"}
        return mixed_label, 80.0
    
    short_label, short_conf = short_comment_analysis(text, rule_score)
    if short_label:
        if return_details:
            return short_label, short_conf, {"method": "short_comment"}
        return short_label, short_conf
    
    bert_label, bert_confidence, bert_scores = advanced_transformer_predict(text)
    
    final_label, final_confidence = advanced_ensemble_predict(
        text, bert_label, bert_confidence, bert_scores, rule_score, emoji_val
    )
    
    final_label, final_confidence = calibrate_confidence(final_label, final_confidence)
    
    if return_details:
        details = {
            "rule_score": rule_score,
            "bert_label": bert_label,
            "final_label": final_label,
            "final_confidence": final_confidence
        }
        return final_label, final_confidence, details
    
    return final_label, final_confidence

def smart_decision_engine(text, v4_label, v4_conf, simple_label, simple_conf):
    word_count = len(text.split())
    
    safe_print(f"   [DECISION] V6={v4_label}({v4_conf:.0f}) vs Simple={simple_label}({simple_conf:.0f})")
    
    if v4_conf < 60 and simple_conf < 60:
        safe_print(f"   [NEUTRAL] BOTH LOW CONFIDENCE: Netral")
        return "Netral", 60
    
    if abs(v4_conf - simple_conf) > 20:
        if v4_conf > simple_conf:
            safe_print(f"   🎯 V6 lebih yakin: {v4_label}")
            return v4_label, v4_conf
        else:
            safe_print(f"   🎯 SIMPLE lebih yakin: {simple_label}")
            return simple_label, simple_conf
    
    if v4_label == simple_label:
        final_conf = max(v4_conf, simple_conf)
        safe_print(f"   [OK] AGREEMENT: {v4_label}")
        return v4_label, final_conf
    
    if v4_conf > 75:
        safe_print(f"   🔥 V6 high confidence: {v4_label}")
        return v4_label, v4_conf
    
    if simple_conf > 75:
        safe_print(f"   🔥 SIMPLE high confidence: {simple_label}")
        return simple_label, simple_conf
    
    safe_print(f"   ⚖️ CONFLICT DEFAULT: Netral")
    return "Netral", 65

# ===============================
# MAIN ANALYZER V6.1 (WITH TIME SERIES)
# ===============================
def analyze_sentiment(text, return_confidence=False, return_details=False, true_label=None, save_to_history=True):
    """
    Optimized Sentiment Analysis V6.1
    Dengan Time Series support untuk grafik harian
    """
    if not isinstance(text, str) or not text.strip():
        if return_details:
            return "Netral", 0.0, {"method": "fallback"}
        elif return_confidence:
            return "Netral", 0.0
        return "Netral"
    
    original_text = text
    
    safe_print(f"\n[ANALYSIS] {text[:80]}...")
    
    # Advanced Engine V6.0
    v4_label, v4_conf, v4_details = advanced_analyze(text, return_details=True)
    safe_print(f"   [BERT] V6.0: {v4_label} ({v4_conf:.1f}%)")
    
    # Simple Model
    simple_label, simple_conf = simple_sentiment(text)
    safe_print(f"   [ROBERTA] SIMPLE: {simple_label} ({simple_conf:.1f}%)")
    
    # Smart Decision
    final_label, final_confidence = smart_decision_engine(
        text, v4_label, v4_conf, simple_label, simple_conf
    )
    
    safe_print(f"   [FINAL] FINAL: {final_label} ({final_confidence:.1f}%)")
    
    # 🔥 SIMPAN KE HISTORY UNTUK TIME SERIES
    if save_to_history:
        add_to_history(original_text, final_label, final_confidence)
    
    # Log error untuk pembelajaran
    if true_label and true_label != final_label:
        with open(ERROR_LOG_PATH, "a", encoding='utf-8') as f:
            f.write(f"{original_text[:100]} | Predicted: {final_label} | True: {true_label}\n")
        
        try:
            os.makedirs(os.path.dirname(TRAINING_DATA_PATH), exist_ok=True)
            with open(TRAINING_DATA_PATH, 'a', encoding='utf-8') as f:
                record = {
                    "text": original_text,
                    "true_label": true_label,
                    "predicted_label": final_label,
                    "v6_label": v4_label,
                    "simple_label": simple_label,
                    "timestamp": datetime.now().isoformat()
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        except:
            pass
        
        try:
            training_data_count = sum(1 for _ in open(TRAINING_DATA_PATH, 'r', encoding='utf-8')) if os.path.exists(TRAINING_DATA_PATH) else 0
            if training_data_count > 200:
                safe_print(f"   🔄 Auto-retraining RF with {training_data_count} samples...")
                train_rf_from_feedback()
        except:
            pass
    
    if return_details:
        details = {
            "v6_label": v4_label,
            "v6_confidence": v4_conf,
            "simple_label": simple_label,
            "simple_confidence": simple_conf,
            "final_label": final_label,
            "final_confidence": final_confidence,
            "method": "optimized_v6.1"
        }
        return final_label, final_confidence, details
    elif return_confidence:
        return final_label, final_confidence
    else:
        return final_label

# ===============================
# TIME SERIES API FUNCTIONS (UNTUK FRONTEND)
# ===============================
def get_trend_data(days=30):
    """
    Dapatkan data trend untuk chart (format yang siap dikirim ke frontend)
    """
    chart_data = format_trend_chart(days)
    return {
        "labels": chart_data["labels"],
        "positif": chart_data["positif"],
        "negatif": chart_data["negatif"],
        "netral": chart_data["netral"],
        "total": chart_data["total"]
    }

def get_trend_summary():
    """
    Dapatkan ringkasan trend
    """
    stats = get_history_stats()
    chart_data = format_trend_chart(7)  # 7 hari terakhir
    
    return {
        "total_analyzed": stats["total"],
        "total_positive": stats["positif"],
        "total_negative": stats["negatif"],
        "total_neutral": stats["netral"],
        "last_7_days": {
            "dates": chart_data["labels"][:7],
            "positive_counts": chart_data["positif"][:7],
            "negative_counts": chart_data["negatif"][:7]
        }
    }

# ===============================
# AUTO RETRAIN RF
# ===============================
def train_rf_from_feedback():
    training_data = []
    if os.path.exists(TRAINING_DATA_PATH):
        with open(TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    training_data.append(json.loads(line))
                except:
                    continue
    
    if len(training_data) < 50:
        safe_print(f"⚠️ Need at least 50 samples for training. Currently: {len(training_data)}")
        return None
    
    safe_print(f"\n📊 Training RF with {len(training_data)} samples...")
    
    X_train = []
    y_train = []
    
    for item in training_data:
        text = item['text']
        true_label = item['true_label']
        
        text_norm = normalize_text(text)
        rule_score, _ = rule_based_score(text_norm)
        emoji_val = emoji_score(text_norm)
        _, _, bert_scores = advanced_transformer_predict(text_norm)
        
        features = rf_validator.extract_features(text_norm, bert_scores, rule_score, emoji_val)
        X_train.append(features)
        y_train.append(LABEL_TO_ID[true_label])
    
    return rf_validator.train(X_train, y_train)

# ===============================
# EVALUATION FUNCTION
# ===============================
def evaluate_model(test_texts, test_labels):
    predictions = []
    for text in test_texts:
        pred, _ = analyze_sentiment(text, return_confidence=True, save_to_history=False)
        predictions.append(pred)
    
    safe_print("\n" + "="*60)
    safe_print("📊 EVALUATION RESULTS")
    safe_print("="*60)
    safe_print(classification_report(test_labels, predictions, target_names=LABELS))
    
    safe_print("\n📊 Confusion Matrix:")
    safe_print(confusion_matrix(test_labels, predictions, labels=LABELS))
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "classification_report": classification_report(test_labels, predictions, target_names=LABELS, output_dict=True),
        "total_samples": len(test_texts)
    }
    
    with open(EVAL_RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results

# ===============================
# UTILITY FUNCTIONS
# ===============================
def analyze_batch(texts, batch_size=16, save_to_history=True):
    results = []
    total = len(texts)
    safe_print(f"📊 Menganalisis {total} teks dalam batch...")
    start_time = time.time()
    
    for i, text in enumerate(texts):
        try:
            result = analyze_sentiment(text, return_confidence=True, save_to_history=save_to_history)
            results.append(result)
        except Exception as e:
            results.append(("Netral", 0.0))
        
        if (i + 1) % 50 == 0:
            safe_print(f"  Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
    
    elapsed = time.time() - start_time
    safe_print(f"✅ Batch analysis selesai dalam {elapsed:.2f}s")
    return results

def get_sentiment_stats(results):
    sentiments = [r[0] for r in results]
    total = len(sentiments)
    return {
        "total": total,
        "positif": sentiments.count("Positif"),
        "negatif": sentiments.count("Negatif"),
        "netral": sentiments.count("Netral"),
        "positif_pct": round(sentiments.count("Positif")/total*100, 1) if total > 0 else 0,
        "negatif_pct": round(sentiments.count("Negatif")/total*100, 1) if total > 0 else 0,
    }

def get_ensemble_info():
    return {
        "version": "6.1",
        "status": "OPTIMIZED WITH TIME SERIES",
        "advanced_model": MODEL_NAME,
        "simple_model": SIMPLE_MODEL_NAME,
        "bert_weight": BERT_WEIGHT,
        "rule_weight": RULE_WEIGHT,
        "rf_weight": RF_WEIGHT,
        "negative_multiplier": NEGATIVE_MULTIPLIER,
        "positive_boost": POSITIVE_BOOST,
        "positive_bias": POSITIVE_BIAS,
        "rf_trained": rf_validator.is_trained,
        "history_count": len(sentiment_history),
        "features": [
            "hard_rules",
            "mixed_sentiment_detection",
            "confidence_calibration",
            "auto_retrain",
            "evaluation_system",
            "time_series_analysis"  # 🔥 BARU
        ]
    }

# ===============================
# TESTING
# ===============================
if __name__ == "__main__":
    safe_print("\n" + "="*60)
    safe_print("🧪 TESTING OPTIMIZED MODEL (V6.1 - TIME SERIES)")
    safe_print("="*60)
    
    test_cases = [
        ("alhamdulillah", "Positif"),
        ("ga jelek kok", "Positif"),
        ("tidak jelek", "Positif"),
        ("parah keren banget", "Positif"),
        ("gila bagus", "Positif"),
        ("keren sih tapi mahal", "Negatif"),
        ("bagus tapi mahal", "Negatif"),
        ("jelek tapi murah", "Positif"),
        ("biasa aja", "Netral"),
        ("lumayan", "Netral"),
        ("anjing", "Negatif"),
        ("keren", "Positif"),
        ("jelek", "Negatif"),
        ("😍", "Positif"),
        ("🤮", "Negatif"),
    ]
    
    correct = 0
    for text, expected in test_cases:
        sentiment, conf, details = analyze_sentiment(text, return_confidence=True, return_details=True)
        status = "✅" if sentiment == expected else "❌"
        safe_print(f"\n{status} '{text}'")
        safe_print(f"   Result: {sentiment} ({conf:.1f}%) | Expected: {expected}")
        if sentiment == expected:
            correct += 1
    
    safe_print(f"\n📊 ACCURACY: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.1f}%)")
    
    # 🔥 TEST TIME SERIES
    safe_print("\n" + "="*60)
    safe_print("📊 TIME SERIES DATA TEST")
    safe_print("="*60)
    
    history_stats = get_history_stats()
    safe_print(f"📈 History stats: {history_stats['total']} total analisis")
    safe_print(f"   Positif: {history_stats['positif']}")
    safe_print(f"   Negatif: {history_stats['negatif']}")
    safe_print(f"   Netral: {history_stats['netral']}")
    
    trend_data = get_trend_data(days=7)
    safe_print(f"\n📊 Trend data (7 hari terakhir):")
    for i, date in enumerate(trend_data["labels"][:7]):
        safe_print(f"   {date}: Positif={trend_data['positif'][i]}, Negatif={trend_data['negatif'][i]}, Netral={trend_data['netral'][i]}")
    
    safe_print(f"\n⚖️  Ensemble Info: {get_ensemble_info()}")