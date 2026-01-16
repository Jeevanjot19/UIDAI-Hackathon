"""Create a sampled dataset for faster model development"""
import pandas as pd
import numpy as np

print("="*80)
print("CREATING SAMPLE DATASET FOR MODEL DEVELOPMENT")
print("="*80)

# Load with sampling
print("\n[LOAD] Reading CSV in chunks...")
chunk_size = 500000
sample_rate = 0.1  # 10% sample

sampled_chunks = []
for i, chunk in enumerate(pd.read_csv('data/processed/aadhaar_with_features.csv', chunksize=chunk_size)):
    # Sample 10% from each chunk
    sampled = chunk.sample(frac=sample_rate, random_state=42)
    sampled_chunks.append(sampled)
    print(f"   Processed chunk {i+1}: {len(chunk):,} rows → {len(sampled):,} sampled")
    
df_sample = pd.concat(sampled_chunks, ignore_index=True)

print(f"\n[SAMPLE] Total sampled: {len(df_sample):,} rows ({len(df_sample)/2947681*100:.1f}% of original)")
print(f"[SAMPLE] Shape: {df_sample.shape}")

# Save sampled dataset
print("\n[SAVE] Saving sampled dataset...")
df_sample.to_csv('data/processed/aadhaar_sample_300k.csv', index=False)
print(f"   ✅ Saved to data/processed/aadhaar_sample_300k.csv")

# Also save as parquet for faster loading
print("\n[SAVE] Saving as parquet...")
df_sample.to_parquet('data/processed/aadhaar_sample_300k.parquet', compression='snappy', index=False)
print(f"   ✅ Saved to data/processed/aadhaar_sample_300k.parquet")

print("\n[DONE] Sample dataset created successfully!")
print(f"        - Original: 2,947,681 rows")
print(f"        - Sample:   {len(df_sample):,} rows ({sample_rate*100:.0f}%)")
print(f"        - File size: Much smaller for faster prototyping")
