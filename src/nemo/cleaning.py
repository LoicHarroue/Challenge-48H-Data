import pandas as pd

# Charger les CSV
train_df = pd.read_csv("src/data/raw/train.csv")
test_df = pd.read_csv("src/data/raw/test.csv")

# Étape 1 : Créer TotalSF
for df in [train_df, test_df]:
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

# Étape 2 : Créer HouseAge et RemodAge
for df in [train_df, test_df]:
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

# Étape 3 : Supprimer les colonnes de date originales
drop_date_cols = ["YearBuilt", "YearRemodAdd", "YrSold"]
train_df.drop(columns=drop_date_cols, inplace=True)
test_df.drop(columns=drop_date_cols, inplace=True)

# Étape 4 : Convertir les booléens
for df in [train_df, test_df]:
    df["CentralAir"] = df["CentralAir"].map({"Y": 1, "N": 0})

# Étape 5 : One-hot encoding des variables catégorielles
train_target = train_df["SalePrice"]
train_df = train_df.drop("SalePrice", axis=1)

# Combiner pour un encodage cohérent
combined = pd.concat([train_df, test_df], axis=0)
combined = pd.get_dummies(combined, drop_first=True)

# Séparer à nouveau
train_encoded = combined.iloc[:len(train_df)].copy()
test_encoded = combined.iloc[len(train_df):].copy()
train_encoded["SalePrice"] = train_target.values

# Sauvegarder les fichiers nettoyés
train_encoded.to_csv("src/nemo/train_cleaned.csv", index=False)
test_encoded.to_csv("src/nemo/test_cleaned.csv", index=False)
