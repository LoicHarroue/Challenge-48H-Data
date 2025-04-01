import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Charger le fichier
df = pd.read_csv('src/data/raw/train.csv')

# 2. FEATURE ENGINEERING
current_year = df["YrSold"].max()
df["HouseAge"] = current_year - df["YearBuilt"]
df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

# 3. Supprimer les colonnes avec trop de valeurs manquantes
missing_thresh = 0.4
missing_ratio = df.isnull().mean()
cols_to_drop = missing_ratio[missing_ratio > missing_thresh].index
df_cleaned = df.drop(columns=cols_to_drop)

# 🔁 4. Sauvegarder uniquement la colonne Id dans un DataFrame
df_id = df_cleaned[["Id"]].reset_index(drop=True)

# 5. Séparer features et target
X = df_cleaned.drop(columns=["Id", "SalePrice"])
y = df_cleaned["SalePrice"]

# 6. Identifier les colonnes numériques et catégorielles
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

# 7. Pipelines
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# 8. ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# 9. Transformer les données
X_processed = preprocessor.fit_transform(X)

# 10. Créer le DataFrame final (hors Id)
cat_feature_names = preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(categorical_cols)
all_feature_names = np.concatenate([numeric_cols, cat_feature_names])
df_final = pd.DataFrame(X_processed, columns=all_feature_names)
df_final["SalePrice"] = y.reset_index(drop=True)

# 🔁 11. Réorganiser avec Id au début
df_final = pd.concat([df_id, df_final], axis=1)

# 12. Afficher un aperçu
print("Shape finale :", df_final.shape)
print(df_final.head())

# 13. Sauvegarder dans un CSV
output_path = "src/data/processed/trainNemo.csv"
df_final.to_csv(output_path, index=False)
print(f"Fichier sauvegardé : {output_path}")
