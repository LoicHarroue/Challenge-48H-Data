import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Charger les données d'entraînement
df = pd.read_csv("../../data/raw/trainNemo.csv")

# Séparer les features et la cible
X = df.drop(columns=["SalePrice", "Id"])
y = df["SalePrice"]

# Identifier les colonnes numériques et catégoriques
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object", "category"]).columns

# Pipelines de transformation
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Transformer les colonnes
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# Appliquer la transformation
X_processed = preprocessor.fit_transform(X)
X_processed = np.array(X_processed)  # Convertir en tableau numpy

# Séparer en train/test
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Initialiser et entraîner CatBoost
catboost_model = CatBoostRegressor(iterations=10000, learning_rate=0.05, depth=10, verbose=100, use_best_model=True, bagging_temperature=2,eval_metric="RMSE")
print("\nTraining CatBoost...")
catboost_model.fit(X_train, y_train, eval_set=(X_test, y_test))

# Évaluer le modèle
y_pred = catboost_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"CatBoost RMSE: {rmse}")

# Charger les nouvelles données
test_df = pd.read_csv("../../data/raw/test.csv")
X_new = test_df.drop(columns=["Id"])

# Appliquer la même transformation aux nouvelles données
X_new_processed = preprocessor.transform(X_new)
X_new_processed = np.array(X_new_processed)

# Prédictions sur les nouvelles données
y_new_pred = catboost_model.predict(X_new_processed)

# Sauvegarder les résultats
submission = pd.DataFrame({"Id": test_df["Id"], "SalePrice": y_new_pred})
submission.to_csv("submission10kbase.csv", index=False)

print("\nPrédictions CatBoost enregistrées dans submission.csv !")