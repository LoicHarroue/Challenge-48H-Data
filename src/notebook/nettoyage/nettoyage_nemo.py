import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Charger le fichier train.csv
df = pd.read_csv("data/raw/train.csv")

# 🎯 FEATURE ENGINEERING
# 1. Ajouter la variable "HouseAge"
current_year = df["YrSold"].max()
df["HouseAge"] = current_year - df["YearBuilt"]

# 2. Ajouter la variable "TotalSF"
df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

# 🧼 NETTOYAGE
# 1. Supprimer les colonnes avec trop de valeurs manquantes (> 40%)
missing_thresh = 0.4
missing_ratio = df.isnull().mean()
cols_to_drop = missing_ratio[missing_ratio > missing_thresh].index
df_cleaned = df.drop(columns=cols_to_drop)

# 2. Séparer features et cible
X = df_cleaned.drop(columns=["Id", "SalePrice"])
y = df_cleaned["SalePrice"]

# 3. Identifier les colonnes numériques et catégorielles
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

# 4. Pipelines
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# 5. ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# 6. Transformation finale
X_processed = preprocessor.fit_transform(X)

# Dimensions du dataset transformé
print("Dimensions du dataset final :", X_processed.shape)