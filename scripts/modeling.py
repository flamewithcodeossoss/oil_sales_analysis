import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

def prepare_features(df):
    # Create copy to avoid modifying original dataframe
    df_prep = df.copy()
    
    # Drop columns that are too specific or contain unique text
    df_prep = df_prep.drop(['store_name', 'sku'], axis=1)
    
    # Convert size from string (e.g., '0.6L') to float (e.g., 0.6)
    df_prep['size'] = df_prep['size'].str.replace('L', '').astype(float)
    
    # Add volume_sales as a feature (it's not circular as it's a different measure)
    numeric_features = ['average_price', 'year', 'month', 'volume_sales', 'size']
    
    # Get categorical columns for one-hot encoding
    categorical_features = ['city', 'brand', 'class', 'manufacturer', 'price_bracket']
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df_prep, columns=categorical_features, drop_first=True)
    
    # Scale numeric features
    scaler = StandardScaler()
    df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])
    
    return df_encoded, numeric_features

def regression_model(df):
    # Prepare features
    df_encoded, numeric_features = prepare_features(df)
    
    # Separate features and target
    y = df['value_sales']
    X = df_encoded.drop(['value_sales'], axis=1)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define model and parameters for tuning
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Make predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    # Calculate metrics
    print('Model Performance Metrics:')
    print('-' * 25)
    print('Training Set:')
    print(f'R² Score: {r2_score(y_train, y_train_pred):.4f}')
    print(f'RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.2f}')
    print(f'MAE: {mean_absolute_error(y_train, y_train_pred):.2f}')
    
    print('\nTest Set:')
    print(f'R² Score: {r2_score(y_test, y_test_pred):.4f}')
    print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.2f}')
    print(f'MAE: {mean_absolute_error(y_test, y_test_pred):.2f}')
    
    # Cross-validation score
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')
    print(f'\nCross-validation R² scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')
    
    # Feature importance
    print('\nTop 10 Most Important Features:')
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
    for _, row in feature_importance.iterrows():
        print(f'{row["feature"]}: {row["importance"]:.4f}')
    
    print(f'\nBest Parameters: {grid_search.best_params_}')
    return model

if __name__ == '__main__':
    df = pd.read_csv('data/oil_sales_assignment_dataset.csv')
    regression_model(df)