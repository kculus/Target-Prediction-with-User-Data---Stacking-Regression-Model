import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
import numpy as np
# Verilerin yüklenmesi
targets_train = pd.read_csv('targets_train.csv')
user_features_test = pd.read_csv("user_features_test.csv")
user_features_train = pd.read_csv("user_features_train.csv")
users_test = pd.read_csv("users_test.csv")
users_train = pd.read_csv("users_train.csv")

# Verilerin birleştirilmesi
train_data = pd.merge(user_features_train, users_train, on="ID")
train_data = pd.merge(targets_train, train_data, on="ID")
test_data = pd.merge(user_features_test, users_test, on="ID")

# Sayısal eksik verilerin doldurulması
train_data['first_prediction'] = train_data['first_prediction'].fillna(train_data['first_prediction'].mean())
duration_columns = [f'Level_{i}_Duration' for i in range(1, 11)]
train_data[duration_columns] =train_data[duration_columns].fillna(train_data[duration_columns].mean())

test_data['first_prediction'] =test_data['first_prediction'].fillna(test_data['first_prediction'].mean())
test_data[duration_columns] =test_data[duration_columns].fillna(test_data[duration_columns].mean())

# Kategorik eksik verilerin doldurulması
categorical_columns = ['country', 'device_brand', 'ad_network']
for column in categorical_columns:
    train_data[column] = train_data[column].fillna('Unknown')
    test_data[column] = test_data[column].fillna('Unknown')

# Özellik mühendisliği: Ekstra özellikler
train_data['Total_Level_Duration'] = train_data[duration_columns].sum(axis=1)
train_data['Avg_Level_Duration'] = train_data[duration_columns].mean(axis=1)

test_data['Total_Level_Duration'] = test_data[duration_columns].sum(axis=1)
test_data['Avg_Level_Duration'] = test_data[duration_columns].mean(axis=1)


# Sayısal olmayan sütunları çıkarıyoruz
train_data = train_data.drop(columns=['first_open_date', 'ID', 'country', 'platform', 'device_category', 'device_brand', 'device_model', 'ad_network'], errors='ignore')
test_data = test_data.drop(columns=['first_open_date', 'ID', 'country', 'platform', 'device_category', 'device_brand', 'device_model', 'ad_network'], errors='ignore')

# Hedef değişken (TARGET) ve özellikler (X)
X_train = train_data.drop(columns=['TARGET'])
y_train = train_data['TARGET']

X_test = test_data

# StandardScaler ile sayısal verilerin ölçeklendirilmesi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model tanımlamaları
linear_model = LinearRegression()
catboost_model = CatBoostRegressor(random_state=42, verbose=0)

# Stacking Regressor ile Linear ve CatBoost birleştiriyoruz
estimators = [
    ('linear', linear_model),
    ('catboost', catboost_model)
]

stacking_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())

# Modeli eğitme
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)
stacking_model.fit(X_train_split, y_train_split)

# Validation seti üzerinde tahmin yapma
y_pred = stacking_model.predict(X_val)

# RMSE metriğini hesaplama
mse = mean_squared_error(y_val, y_pred, squared=False)
print(f'Validation MSE: {mse}')

# Test seti üzerinde tahmin yapma
test_predictions = stacking_model.predict(X_test_scaled)

# Tahmin sonuçlarını CSV dosyasına kaydetme
submission = pd.DataFrame({
    'ID': users_test['ID'],
    'TARGET': test_predictions
})
submission.to_csv('submission.csv', index=False)
