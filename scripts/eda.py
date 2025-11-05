import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def summarize_by(df, group_cols):
    return df.groupby(group_cols).agg({'value_sales': 'sum', 'volume_sales': 'sum'}).reset_index()

def plot_top_brands(df):
    brand_sales = df.groupby('brand')['value_sales'].sum().sort_values(ascending=False)
    brand_sales.head(10).plot(kind='bar')
    plt.title('Top 10 Brands by Sales')
    plt.show()

def plot_sales_trends(df):
    trend = df.groupby(['year', 'month'])['value_sales'].sum().reset_index()
    sns.lineplot(data=trend, x='month', y='value_sales', hue='year', marker='o')
    plt.title('Monthly Sales Trends')
    plt.show()

def find_anomalies(df, col):
    # Show rows with extreme values
    return df[df[col] > df[col].quantile(0.99)]

if __name__ == '__main__':
    df = pd.read_csv('data/cleaned_data.csv')
    print("Top brands by sales:")
    print(df.groupby('brand')['value_sales'].sum().sort_values(ascending=False).head(10))
    plot_top_brands(df)
    plot_sales_trends(df)
    plt.savefig('outputs/sales_trends.png')      # Save plots to outputs/
    print("Plots saved to outputs/")
    print(find_anomalies(df, 'average_price'))