import pandas as pd
import numpy as np
import plotly.express as px
from matplotlib import pyplot as plt
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
df = pd.read_csv("../../data/raw/train.csv")

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
catboost_model = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=10, verbose=100, use_best_model=True, bagging_temperature=2,eval_metric="RMSE")
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

feature_names = preprocessor.get_feature_names_out()
feature_importance = catboost_model.get_feature_importance()

# Vérifier les longueurs
print(f"Taille de feature_importance: {len(feature_importance)}")
print(f"Taille de feature_names: {len(feature_names)}")

# Créer le DataFrame correctement
feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

print(feature_importance_df)

top_22_features = feature_importance_df[:22]

# Créer le graphique interactif avec Plotly
fig = px.bar(
    top_22_features,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 22 des variables les plus importantes dans CatBoost",
    color="Importance",
    color_continuous_scale="blues"
)

fig.update_layout(yaxis=dict(autorange="reversed"))  # Inverser l'axe Y
fig.show()