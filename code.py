import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load dataset
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

# 2. Preprocess data
def preprocess_data(df):
    # Drop rows with missing target
    df = df.dropna(subset=['price'])
    
    # Separate features and target
    X = df.drop(columns=['price'])
    y = df['price']

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Preprocessing pipelines
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor, X, y

# 3. Build and train model
def train_model(X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create pipeline: preprocessing + model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Train
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")

    return model

# 4. Predict on new data
def predict_price(model, new_data):
    # new_data: pandas DataFrame with same features as training
    predicted_prices = model.predict(new_data)
    return predicted_prices

# Example usage
if __name__ == "__main__":
    # Assume 'cars.csv' has columns: 'price', 'brand', 'model', 'year', 'mileage', etc.
    df = load_data('cars.csv')
    preprocessor, X, y = preprocess_data(df)
    model = train_model(X, y, preprocessor)

    # Predict on a few new samples
    sample_data = pd.DataFrame([
        {'brand': 'Toyota', 'model': 'Corolla', 'year': 2018, 'mileage': 25000},
        {'brand': 'BMW', 'model': '3 Series', 'year': 2016, 'mileage': 40000}
    ])
    preds = predict_price(model, sample_data)
    print("Predicted Prices:", preds)
