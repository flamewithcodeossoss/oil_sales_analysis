import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path)
    return df

def handle_missing_values(df):
    # Fill categoricals, impute numerics
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(method='ffill', inplace=True)
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].median(), inplace=True)
    return df

def handle_outliers(df, cols):
    # IQR method
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower, upper)
    return df

def clean_data(path):
    df = load_data(path)
    df = handle_missing_values(df)
    df = handle_outliers(df, ['value_sales', 'volume_sales', 'average_price'])
    df.drop_duplicates(inplace=True)
    return df

if __name__ == '__main__':
    df = clean_data(r"C:\Users\Osama Mohamed Naguib\Desktop\oil_sales_analysis\data\oil_sales_assignment_dataset.csv")
    print(df.head())              # Print a preview of cleaned data
    print(df.describe())          # Print summary stats
    df.to_csv('data/cleaned_data.csv', index=False)  # Save cleaned data
    print('Cleaned data saved to data/cleaned_data.csv')