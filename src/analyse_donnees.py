import pandas as pd
import numpy as np
import plotly as py
import plotly.express as px
import plotly.figure_factory as ff
import networkx as nx
import plotly.graph_objects as go

df = pd.read_csv('data/processed/test.csv')

# Convertir les variables catégorielles en numériques si nécessaire
df_encoded = pd.get_dummies(df)

corr_matrix = df_encoded.corr()

# Créer une heatmap avec Plotly Express
fig = px.imshow(corr_matrix,
                labels=dict(color="Corrélation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale='RdBu_r')

# Afficher la figure
fig.show()
