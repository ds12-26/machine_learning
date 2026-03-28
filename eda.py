# EDA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("../data/features_3_sec.csv")
df = df.drop(columns=['length'])
print(df.shape)

# 1: Univariate — Genre Distribution 
plt.figure(figsize=(12, 5))
sns.countplot(data=df, x='label', order=df['label'].value_counts().index, palette='muted')
plt.title('Genre Distribution (Univariate)')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 2: Bivariate — Tempo by Genre 
plt.figure(figsize=(12, 5))
sns.boxplot(data=df, x='label', y='tempo', palette='muted')
plt.title('Tempo by Genre (Bivariate)')
plt.xlabel('Genre')
plt.ylabel('Tempo (BPM)')
plt.tight_layout()
plt.show()

# 3: Bivariate — Spectral Centroid by Genre 
plt.figure(figsize=(12, 5))
sns.boxplot(data=df, x='label', y='spectral_centroid_mean', palette='muted')
plt.title('Spectral Centroid Mean by Genre (Bivariate)')
plt.xlabel('Genre')
plt.ylabel('Spectral Centroid Mean (Hz)')
plt.tight_layout()
plt.show()

# 4: Multivariate — Correlation Heatmap of Top Features
# Find most discriminative features by variance across genres
feature_variance = df.groupby('label').mean(numeric_only=True).T.var(axis=1).sort_values(ascending=False)
top_features = feature_variance.head(8).index.tolist()
print("Top 8 most discriminative features:", top_features)

plt.figure(figsize=(10, 7))
sns.heatmap(df[top_features].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Top 8 Most Discriminative Features (Multivariate)')
plt.tight_layout()
plt.show()

#5: Multivariate — MFCC 1 vs MFCC 2 by Genre 
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x='mfcc1_mean', y='mfcc2_mean', hue='label', alpha=0.4, palette='tab10')
plt.title('MFCC 1 vs MFCC 2 by Genre (Multivariate)')
plt.xlabel('MFCC 1 Mean')
plt.ylabel('MFCC 2 Mean')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
