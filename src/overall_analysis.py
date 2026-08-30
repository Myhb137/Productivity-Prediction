import pandas as pd 

def overall_analysis(data):
    
    stats = data.describe()
    print("Basic Statistics:\n", stats)

    missing_values = data.isnull().sum()
    print("\nMissing Values:\n", missing_values)

    correlation_matrix = data.corr(numeric_only=True)
    print("\nCorrelation Matrix:\n", correlation_matrix)

    return stats, missing_values, correlation_matrix