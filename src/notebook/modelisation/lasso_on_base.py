import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error

# 1. Chargement et prétraitement des données d'entraînement
df = pd.read_csv("../../data/raw/train.csv")

# Séparer les features et la cible
X = df.drop(columns=["SalePrice", "Id"])
y = df["SalePrice"]

# Identifier les colonnes numériques et catégoriques
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object", "category"]).columns

# Pipeline pour les variables numériques : imputation par la médiane
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# Pipeline pour les variables catégoriques : imputation et OneHotEncoder
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Combiner les pipelines dans un transformateur
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# Appliquer la transformation sur l'ensemble des données
X_processed = preprocessor.fit_transform(X)
X_processed = np.array(X_processed)  # S'assurer que c'est un tableau numpy

# Diviser les données en ensembles d'entraînement et de test (80/20)
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# 2. Entraînement des modèles Ridge et Lasso
ridge_model = Ridge(alpha=1.0, max_iter=100000)
lasso_model = Lasso(alpha=1.0, max_iter=100000)

print("Entraînement de Ridge...")
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_val)
mae_ridge = mean_absolute_error(y_val, y_pred_ridge)
print(f"Ridge MAE: {mae_ridge}\n")

print("Entraînement de Lasso...")
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_val)
mae_lasso = mean_absolute_error(y_val, y_pred_lasso)
print(f"Lasso MAE: {mae_lasso}\n")

# 3. Prédictions sur les données du fichier test.csv
# Charger les données de test
test_df = pd.read_csv("../../data/raw/test.csv")
X_new = test_df.drop(columns=["Id"])

# Appliquer la transformation préalablement apprise sur train.csv
X_new_processed = preprocessor.transform(X_new)
X_new_processed = np.array(X_new_processed)

# Prédire avec Ridge
y_new_pred_ridge = ridge_model.predict(X_new_processed)
submission_ridge = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": y_new_pred_ridge
})
submission_ridge.to_csv("submission_ridge_base.csv", index=False)
print("Prédictions avec Ridge enregistrées dans submission_ridge.csv")

# Prédire avec Lasso
y_new_pred_lasso = lasso_model.predict(X_new_processed)
submission_lasso = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": y_new_pred_lasso
})
submission_lasso.to_csv("submission_lasso_base.csv", index=False)
print("Prédictions avec Lasso enregistrées dans submission_lasso.csv")
