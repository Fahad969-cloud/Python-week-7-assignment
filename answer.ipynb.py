# Task 1: Load and Explore the Dataset

import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
df = pd.read_csv(url, names=columns)
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset structure and info:")
print(df.info())
print("\nMissing values in the dataset:")
print(df.isnull().sum())
df = df.fillna(df.mean())
print("\nDataset after cleaning (if applicable):")
print(df.head())

# Task 2: Basic Data Analysis

print("Basic statistics of numerical columns:")
print(df.describe())


grouped_data = df.groupby('Species').mean()
print("\nMean values for each group (grouped by Species):")
print(grouped_data)


print("\nInteresting findings:")
for column in df.select_dtypes(include='float64').columns:
    max_group = grouped_data[column].idxmax()
    print(f"{column}: {max_group} has the highest mean value ({grouped_data[column].max():.2f})")

# Task 3:Data Visualization
# a. 
import matplotlib.pyplot as plt
import pandas as pd


data = pd.DataFrame({
    'Year': [2018, 2019, 2020, 2021, 2022],
    'Sales': [150, 200, 250, 300, 350]
})

plt.figure(figsize=(10, 6))
plt.plot(data['Year'], data['Sales'], marker='o', color='b', label='Sales Trend', linestyle='-', linewidth=2)


plt.title('Sales Trend Over Time', fontsize=16, fontweight='bold', color='darkblue')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Sales ($)', fontsize=14)
plt.xticks(data['Year'], rotation=45)  # Rotate x-axis labels for readability
plt.yticks(range(0, 401, 50))
plt.legend(title='Legend', loc='upper left', fontsize=12, shadow=True, fancybox=True)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()  # Ensures that labels don't overlap
plt.show()

# b. 
import seaborn as sns

# Example data (replace with your actual data)
data = pd.DataFrame({
    'Species': ['Setosa', 'Versicolor', 'Virginica'],
    'Avg_Petal_Length': [1.4, 4.2, 5.5]
})

plt.figure(figsize=(10, 6))
sns.barplot(x='Species', y='Avg_Petal_Length', data=data, palette='Blues')

# Customizing the plot with titles, axis labels, and grid
plt.title('Average Petal Length per Species', fontsize=16, fontweight='bold', color='darkgreen')
plt.xlabel('Species', fontsize=14)
plt.ylabel('Average Petal Length (cm)', fontsize=14)
plt.xticks(rotation=45)  # Rotate x-axis labels if needed
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# c 

data = pd.Series([5.1, 5.4, 5.3, 5.0, 5.2, 5.8, 5.9, 5.6, 5.3, 5.7])

plt.figure(figsize=(10, 6))
plt.hist(data, bins=5, color='green', edgecolor='black')


plt.title('Distribution of Petal Length', fontsize=16, fontweight='bold', color='darkred')
plt.xlabel('Petal Length (cm)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# d 

data = pd.DataFrame({
    'Sepal_Length': [5.1, 5.9, 5.4, 5.8, 5.5, 6.2, 5.7, 6.0],
    'Petal_Length': [1.4, 4.2, 3.9, 4.0, 4.1, 4.5, 4.3, 4.2]
})

plt.figure(figsize=(10, 6))
plt.scatter(data['Sepal_Length'], data['Petal_Length'], color='r', label='Sepal vs. Petal Length', edgecolor='black')


plt.title('Sepal Length vs Petal Length', fontsize=16, fontweight='bold', color='navy')
plt.xlabel('Sepal Length (cm)', fontsize=14)
plt.ylabel('Petal Length (cm)', fontsize=14)
plt.legend(title='Legend', loc='upper left', fontsize=12, shadow=True)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
