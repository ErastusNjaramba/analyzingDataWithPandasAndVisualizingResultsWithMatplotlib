import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# Set seaborn style for better visualization
sns.set_style("whitegrid")

def load_and_explore_data():
    """Load and explore the Iris dataset"""
    try:
        # Load Iris dataset
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        
        # Display first few rows
        print("\nFirst 5 rows of the dataset:")
        print(df.head())
        
        # Display data info
        print("\nDataset Info:")
        print(df.info())
        
        # Check for missing values
        print("\nMissing Values:")
        print(df.isnull().sum())
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_data(df):
    """Perform basic data analysis"""
    try:
        # Basic statistics
        print("\nBasic Statistics:")
        print(df.describe())
        
        # Group by species and calculate means
        print("\nMean measurements by species:")
        print(df.groupby('species').mean())
        
        # Additional analysis: correlation matrix
        print("\nCorrelation Matrix:")
        print(df.iloc[:, :-1].corr())
        
    except Exception as e:
        print(f"Error in analysis: {e}")

def create_visualizations(df):
    """Create four different visualizations"""
    try:
        # 1. Line chart: Mean measurements across species
        plt.figure(figsize=(10, 6))
        species_means = df.groupby('species').mean()
        for column in species_means.columns:
            plt.plot(species_means.index, species_means[column], marker='o', label=column)
        plt.title('Mean Measurements by Species')
        plt.xlabel('Species')
        plt.ylabel('Measurement (cm)')
        plt.legend()
        plt.savefig('line_chart.png')
        plt.close()
        
        # 2. Bar chart: Average sepal length by species
        plt.figure(figsize=(8, 6))
        sns.barplot(x='species', y='sepal length (cm)', data=df)
        plt.title('Average Sepal Length by Species')
        plt.xlabel('Species')
        plt.ylabel('Sepal Length (cm)')
        plt.savefig('bar_chart.png')
        plt.close()
        
        # 3. Histogram: Sepal length distribution
        plt.figure(figsize=(8, 6))
        plt.hist(df['sepal length (cm)'], bins=20, edgecolor='black')
        plt.title('Distribution of Sepal Length')
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Frequency')
        plt.savefig('histogram.png')
        plt.close()
        
        # 4. Scatter plot: Sepal length vs Petal length
        plt.figure(figsize=(8, 6))
        for species in df['species'].unique():
            species_data = df[df['species'] == species]
            plt.scatter(species_data['sepal length (cm)'], 
                       species_data['petal length (cm)'], 
                       label=species, 
                       alpha=0.6)
        plt.title('Sepal Length vs Petal Length')
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Petal Length (cm)')
        plt.legend()
        plt.savefig('scatter_plot.png')
        plt.close()
        
    except Exception as e:
        print(f"Error in visualization: {e}")

def main():
    """Main function to execute the analysis"""
    print("Iris Dataset Analysis")
    print("="*50)
    
    # Load and explore data
    df = load_and_explore_data()
    
    if df is not None:
        # Perform analysis
        analyze_data(df)
        
        # Create visualizations
        create_visualizations(df)
        
        # Print findings
        print("\nFindings and Observations:")
        print("- The dataset contains 150 samples with no missing values.")
        print("- Setosa species tends to have smaller measurements overall.")
        print("- Virginica species generally has the largest measurements.")
        print("- There's a strong positive correlation between petal length and petal width.")
        print("- Sepal length and petal length show clear separation between species in the scatter plot.")
        
    else:
        print("Analysis aborted due to data loading error.")

if __name__ == "__main__":
    main()