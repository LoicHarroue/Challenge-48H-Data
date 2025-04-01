import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Charger les données d'entraînement
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

# Pipeline pour les variables catégoriques : imputation puis OneHotEncoder
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
X_processed = np.array(X_processed)  # S'assurer que le format est bien un tableau numpy

# Diviser en ensembles d'entraînement et de test (par exemple 80/20)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Initialiser le modèle Random Forest
rf_model = RandomForestRegressor(
    n_estimators=5000,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=2,
    max_features='log2',
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)
# Entraîner le modèle
rf_model.fit(X_train, y_train)

# Prédire sur l'ensemble de test
y_pred = rf_model.predict(X_test)

# Calculer l'erreur absolue moyenne (MAE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"CatBoost RMSE: {rmse}")

# Si vous souhaitez appliquer le modèle à de nouvelles données (exemple avec test.csv)
# Charger les nouvelles données
test_df = pd.read_csv("../../data/raw/test.csv")
X_new = test_df.drop(columns=["Id"])

# Appliquer la transformation préalablement apprise
X_new_processed = preprocessor.transform(X_new)
X_new_processed = np.array(X_new_processed)

# Prédire les prix pour les nouvelles données
y_new_pred = rf_model.predict(X_new_processed)

# Sauvegarder les prédictions dans un fichier CSV
submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": y_new_pred
})
submission.to_csv("submission_forest.csv", index=False)
print("Prédictions enregistrées dans submission.csv")
