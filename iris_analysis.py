# --------------------------------------------------------
# Assignment: Analyzing the Iris Dataset with Pandas and Matplotlib
# --------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# --------------------------------------------------------
# Task 1: Load and Explore the Dataset
# --------------------------------------------------------

# Load Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame
df['species'] = df['target'].map(dict(enumerate(iris.target_names)))

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# --------------------------------------------------------
# Task 2: Basic Data Analysis
# --------------------------------------------------------

print("\nSummary Statistics:")
print(df.describe())

# Grouping: mean petal length per species
grouped = df.groupby("species")["petal length (cm)"].mean()
print("\nAverage Petal Length by Species:")
print(grouped)

# Which species has the longest average petal length?
max_species = grouped.idxmax()
print(f"\nSpecies with longest average petal length: {max_species}")

# --------------------------------------------------------
# Task 3: Data Visualization
# --------------------------------------------------------

# 1. Line Chart (Petal Length trend by sample index)
plt.figure(figsize=(8,5))
plt.plot(df.index, df["petal length (cm)"], label="Petal Length", color="blue")
plt.title("Petal Length Trend Across Samples")
plt.xlabel("Sample Index")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.grid(True)
plt.show()

# 2. Bar Chart (Average Petal Length per Species)
plt.figure(figsize=(8,5))
grouped.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram (Distribution of Sepal Length)
plt.figure(figsize=(8,5))
plt.hist(df["sepal length (cm)"], bins=15, color="green", edgecolor="black")
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot (Sepal Length vs Petal Length by Species)
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="sepal length (cm)", y="petal length (cm)", hue="species", palette="Set1")
plt.title("Sepal Length vs Petal Length by Species")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

# --------------------------------------------------------
# Observations / Findings
# --------------------------------------------------------
print("\nObservations:")
print("1. Virginica species has the longest average petal length.")
print("2. Sepal length distribution is slightly right-skewed, mostly between 5 and 6 cm.")
print("3. Setosa species is clearly separable from the others in scatter plot.")
print("4. Versicolor and Virginica show more overlap in petal/sepal lengths.")
