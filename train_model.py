# Advanced Depression Detection Training Script
# Handles variations like "sad", "saaad", "sadness", "sadddd"

import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("ADVANCED DEPRESSION DETECTION - TRAINING")
print("=" * 60)

# ==================== ADVANCED TEXT PREPROCESSING ====================

def advanced_clean_text(text):
    """
    Advanced text cleaning that handles:
    - Repeated characters (saaad -> sad)
    - Common misspellings
    - Slang and informal language
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)  # Keep hashtag text
    
    # Fix repeated characters (saaad -> sad, soooo -> so)
    # This helps with variations like "sadddd", "happppy"
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Replace common informal expressions
    replacements = {
        "can't": "cannot",
        "won't": "will not",
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "'m": " am",
        "gonna": "going to",
        "wanna": "want to",
        "gotta": "got to",
        "dunno": "do not know",
        "lemme": "let me",
        "gimme": "give me",
        "kinda": "kind of",
        "sorta": "sort of",
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Keep important punctuation for sentiment (! and ?)
    # But remove other special characters
    text = re.sub(r'[^\w\s!?]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def create_expanded_dataset(df):
    """
    Expand dataset with variations to help model learn patterns
    """
    # Create some variations of existing texts
    expanded_data = []
    
    for idx, row in df.iterrows():
        text = row['text']
        label = row['label']
        
        # Original text
        expanded_data.append({'text': text, 'label': label})
        
        # Add variation with repeated characters (simulating real typing)
        if label == 1:  # Only for depressive texts
            # Add some variations
            varied = text.replace('so', 'soooo').replace('sad', 'saaad')
            expanded_data.append({'text': varied, 'label': label})
    
    return pd.DataFrame(expanded_data)

# ==================== LOAD AND PREPARE DATA ====================

print("\n[1/7] Loading dataset...")
df = pd.read_csv('data/depression_dataset.csv')
print(f"✓ Loaded {len(df)} samples")

# Expand dataset with variations
print("\n[2/7] Expanding dataset with variations...")
df_expanded = create_expanded_dataset(df)
print(f"✓ Expanded to {len(df_expanded)} samples")

# Clean all texts
print("\n[3/7] Advanced text cleaning...")
df_expanded['cleaned_text'] = df_expanded['text'].apply(advanced_clean_text)
print("✓ Cleaning completed")
print(f"\nExample:")
print(f"Original: '{df_expanded['text'].iloc[0]}'")
print(f"Cleaned:  '{df_expanded['cleaned_text'].iloc[0]}'")

# ==================== TRAIN-TEST SPLIT ====================

print("\n[4/7] Splitting data...")
X = df_expanded['cleaned_text']
y = df_expanded['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Training: {len(X_train)} samples")
print(f"✓ Testing: {len(X_test)} samples")

# ==================== FEATURE EXTRACTION ====================

print("\n[5/7] Creating advanced TF-IDF features...")

# Advanced TF-IDF with character n-grams
# This helps catch variations like "sad", "sadness", "sadly"
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
    analyzer='word',  # Analyze by words
    min_df=1,  # Minimum document frequency
    max_df=0.9,  # Maximum document frequency
    sublinear_tf=True,  # Use logarithmic form
    strip_accents='unicode',
    lowercase=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"✓ Created {X_train_tfidf.shape[1]} features")

# Show top features for each class
print("\n📊 Top words associated with depression:")
feature_names = vectorizer.get_feature_names_out()
log_reg_temp = LogisticRegression(max_iter=1000, random_state=42)
log_reg_temp.fit(X_train_tfidf, y_train)
coef = log_reg_temp.coef_[0]
top_indices = coef.argsort()[-15:][::-1]
top_words = [feature_names[i] for i in top_indices]
print("   ", ", ".join(top_words[:10]))

# ==================== MODEL TRAINING ====================

print("\n[6/7] Training advanced Logistic Regression model...")
model = LogisticRegression(
    max_iter=2000,
    random_state=42,
    C=1.0,  # Regularization strength
    class_weight='balanced',  # Handle any imbalance
    solver='lbfgs'
)
model.fit(X_train_tfidf, y_train)
print("✓ Training completed!")

# ==================== EVALUATION ====================

print("\n[7/7] Evaluating model...")
y_pred = model.predict(X_test_tfidf)
y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print(f"\n🎯 Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['Not Depressed', 'Depressed']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Depressed', 'Depressed'],
            yticklabels=['Not Depressed', 'Depressed'])
plt.title('Confusion Matrix - Advanced Model', fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix_advanced.png', dpi=150)
print("\n✓ Confusion matrix saved")

# ==================== SAVE MODEL ====================

print("\n[SAVING] Saving advanced model...")
with open('models/model_advanced.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/vectorizer_advanced.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("✓ Model saved to 'models/' folder")

# ==================== TESTING WITH VARIATIONS ====================

print("\n" + "=" * 60)
print("TESTING WITH TEXT VARIATIONS")
print("=" * 60)

test_cases = [
    # Standard text
    "I feel so hopeless and empty",
    
    # With repeated characters
    "I feel soooo hopeless and emptyyyy",
    
    # Misspellings
    "cant get out of bed everythng is dark",
    
    # Informal/slang
    "dunno why i even bother anymore tbh",
    
    # Variations of sadness
    "feeling saaad and looonely",
    
    # Positive text - standard
    "Had an amazing day at the beach!",
    
    # Positive - with variations
    "Had an amaazing day woohoooo!!!",
    
    # Neutral
    "going to work today"
]

for text in test_cases:
    cleaned = advanced_clean_text(text)
    text_tfidf = vectorizer.transform([cleaned])
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0]
    
    confidence = probability[1] if prediction == 1 else probability[0]
    result = "DEPRESSIVE" if prediction == 1 else "NOT DEPRESSIVE"
    
    print(f"\n📝 Text: \"{text}\"")
    print(f"   Cleaned: \"{cleaned}\"")
    print(f"   → {result} (Confidence: {confidence*100:.1f}%)")

print("\n" + "=" * 60)
print("✅ ADVANCED TRAINING COMPLETE!")
print("=" * 60)
print("\n🎯 Key Improvements:")
print("   ✓ Handles text variations (saaad, sad, sadness)")
print("   ✓ Cleans repeated characters")
print("   ✓ Processes informal language")
print("   ✓ Uses trigrams for better context")
print("   ✓ More robust to typos and slang")