# Script to Download and Use Real Mental Health Dataset
# Dataset: Reddit Mental Health (Depression vs SuicideWatch vs Normal)

import pandas as pd
import numpy as np
import requests
import io

print("=" * 70)
print("DOWNLOADING REAL MENTAL HEALTH DATASET")
print("=" * 70)

# ==================== DOWNLOAD DATASET ====================

print("\n[1/3] Downloading Reddit Mental Health Dataset...")
print("Source: Kaggle - Reddit Mental Health Posts")

# Multiple dataset sources (try each until one works)
dataset_urls = [
    # Option 1: Direct GitHub dataset
    "https://raw.githubusercontent.com/ajaymuktha/Depression-Detection/main/data/depression_dataset_reddit_cleaned.csv",
    
    # Option 2: Backup dataset
    "https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Data/classification_sprint/Tweets.csv"
]

df = None
for i, url in enumerate(dataset_urls, 1):
    try:
        print(f"\nTrying source {i}...")
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            print(f"✓ Successfully downloaded from source {i}!")
            break
    except Exception as e:
        print(f"✗ Source {i} failed: {e}")
        continue

if df is None:
    print("\n⚠️  Automatic download failed. Using manual download instructions...")
    print("\nMANUAL DOWNLOAD INSTRUCTIONS:")
    print("=" * 70)
    print("1. Go to: https://www.kaggle.com/datasets/infamouscoder/mental-health-social-media")
    print("2. Click 'Download' button")
    print("3. Extract the ZIP file")
    print("4. Move 'Mental-Health-Twitter.csv' to your 'data' folder")
    print("5. Rename it to 'mental_health_dataset.csv'")
    print("6. Run this script again")
    print("=" * 70)
    exit()

# ==================== PROCESS DATASET ====================

print(f"\n[2/3] Processing dataset...")
print(f"Original shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Check what columns we have and adapt
if 'text' in df.columns and 'label' in df.columns:
    # Already in correct format
    pass
elif 'post_text' in df.columns:
    df = df.rename(columns={'post_text': 'text'})
elif 'tweet' in df.columns:
    df = df.rename(columns={'tweet': 'text'})
elif 'message' in df.columns:
    df = df.rename(columns={'message': 'text'})

# Handle different label formats
if 'label' not in df.columns:
    if 'sentiment' in df.columns:
        # Convert sentiment to binary
        df['label'] = df['sentiment'].apply(lambda x: 1 if 'negative' in str(x).lower() or 'depression' in str(x).lower() else 0)
    elif 'subreddit' in df.columns:
        # Convert subreddit to binary (depression subreddits = 1, others = 0)
        depression_subs = ['depression', 'suicidewatch', 'mentalhealth', 'anxiety']
        df['label'] = df['subreddit'].apply(lambda x: 1 if any(sub in str(x).lower() for sub in depression_subs) else 0)

# Clean the data
print("\nCleaning data...")
df = df[['text', 'label']].copy()
df = df.dropna()
df = df[df['text'].str.len() > 10]  # Remove very short posts
df['text'] = df['text'].str[:500]  # Limit length

# Balance the dataset
print("\nBalancing dataset...")
depressed = df[df['label'] == 1]
not_depressed = df[df['label'] == 0]

# Sample equal amounts
min_samples = min(len(depressed), len(not_depressed))
min_samples = min(min_samples, 5000)  # Cap at 5000 per class

depressed_sample = depressed.sample(n=min_samples, random_state=42)
not_depressed_sample = not_depressed.sample(n=min_samples, random_state=42)

df_balanced = pd.concat([depressed_sample, not_depressed_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✓ Final dataset: {len(df_balanced)} samples")
print(f"  - Depressed: {sum(df_balanced['label'] == 1)}")
print(f"  - Not Depressed: {sum(df_balanced['label'] == 0)}")

# ==================== SAVE DATASET ====================

print(f"\n[3/3] Saving processed dataset...")

# Save to data folder
import os
os.makedirs('data', exist_ok=True)
df_balanced.to_csv('data/mental_health_dataset.csv', index=False)

print("✓ Dataset saved to 'data/mental_health_dataset.csv'")

# Show sample
print("\n" + "=" * 70)
print("SAMPLE DATA")
print("=" * 70)
print("\nDepressed examples:")
print(df_balanced[df_balanced['label'] == 1].head(3)['text'].values)
print("\nNot Depressed examples:")
print(df_balanced[df_balanced['label'] == 0].head(3)['text'].values)

print("\n" + "=" * 70)
print("✅ DATASET READY!")
print("=" * 70)
print("\nNext step: Run the training script")
print("Command: python train_model_real_data.py")