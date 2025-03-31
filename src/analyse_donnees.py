import pandas as pd
import numpy as np
import plotly as py
import plotly.express as px
import plotly.figure_factory as ff
import networkx as nx
import plotly.graph_objects as go

df = pd.read_csv('data/test.csv')

# Convertir les variables catégorielles en numériques si nécessaire
df_encoded = pd.get_dummies(df)

seuil = 0.5

# Calculer la matrice de corrélation
corr_matrix = df_encoded.corr()

# Filtrer les corrélations fortes
edges = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > seuil:
            edges.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

# Construire le graphe
G = nx.Graph()
G.add_weighted_edges_from(edges)

# Positionnement des nœuds
pos = nx.spring_layout(G)

# Création des nœuds pour Plotly
node_x, node_y, node_labels = [], [], []
for node, (x, y) in pos.items():
    node_x.append(x)
    node_y.append(y)
    node_labels.append(node)

# Création des arêtes pour Plotly
edge_x, edge_y, edge_widths = [], [], []
for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    edge_widths.append(abs(edge[2]['weight']) * 5)

# Tracer les arêtes
edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=2, color='gray'),
    mode='lines')

# Tracer les nœuds
node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=node_labels,
    textposition='top center',
    marker=dict(size=10, color='blue')
)

# Création de la figure
fig = go.Figure(data=[edge_trace, node_trace])
fig.update_layout(showlegend=False, title='Graph Réseau des Corrélations')

# Affichage
fig.show()
