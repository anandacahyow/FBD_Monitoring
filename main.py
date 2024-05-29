import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

def load_data(file):
    columns = ['Date On Shift', 'Time', 'Shift Name', 'Recipe Name', 'Operator Name', 'FLM  Name',
               'TR6701 Temp. 1', 'TR6702 Temp. 2', 'TR6702 Temp. 3', 'TR6702 Temp. 4', 'TR6705 Ex. Temp.',
               'PT6701 Press. 1', 'PT6702 Press. 2', 'PT6703 Press. 3', 'PT6704 Press. 4',
               'DP6701 Diff Press. 1', 'DP6702 Diff Press. 2', 'DP6703 Diff Press. 3', 'DP6704 Diff Press. 4',
               'TT6701 Mix. Temp. 1', 'TT6702 Mix. Temp. 2', 'TT6703 Mix. Temp. 3', 'TT6704 Mix. Temp. 4',
               'PT6711 Mix. Press. 1', 'PT6712 Mix. Press. 2', 'M6701 Speed Exh', 'M6702 Speed Blwr 1',
               'M6703 Speed Blwr 2', 'M6704 Speed Blwr 3', 'M6705 Speed Blwr 4', 'M6706 Speed FBD Vibr',
               'IS_SteamSupply', 'IS_SteamFlow', 'IS_SteamFlowTOT']
    
    data = load_data(file, columns)
    data['Datetime'] = pd.to_datetime(data['Date On Shift'] + ' ' + data['Time'])
    
    for i in range(1, len(data)):
        if data.loc[i, 'Datetime'] < data.loc[i - 1, 'Datetime']:
            data.loc[i, 'Datetime'] += timedelta(days=1)
            
    data = data.iloc[-2500:, :]
    data = data[['Datetime'] + [col for col in data.columns if col not in ['Date On Shift', 'Time']]]
    return data

def preprocess_data(data):
    data_index = data.iloc[:, :4]
    data = data.iloc[:, 5:-2]
    
    numeric_cols = data.columns
    for col in numeric_cols:
        quartile = data[col].quantile([0.25, 0.5, 0.75])
        data[col].fillna(quartile[0.5], inplace=True)
    
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    return data, data_index, numeric_cols

def create_correlation_heatmap(data):
    correlation_matrix = data.corr()
    fig_corr_heatmap = px.imshow(correlation_matrix,
                                 labels=dict(x="Numerical Columns", y="Numerical Columns"))
    fig_corr_heatmap.update_layout(title="Correlation Heatmap")
    st.plotly_chart(fig_corr_heatmap)

def apply_pca(data, n_components):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    
    return pca_result

def plot_clusters(algorithm, data, pca_result, numeric_cols, data_index):
    cluster_labels = algorithm.labels_
    data.insert(0, 'Cluster', cluster_labels)
    
    combined_data = pd.concat([data_index, data], axis=1)
    
    st.dataframe(combined_data)

def main():
    st.title('Your App Title')
    st.sidebar.title('Upload CSV File')
    
    file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
    
    if file is not None:
        data = load_data(file)
        st.write("Data loaded successfully!")
        
        if st.sidebar.button("Preprocess Data"):
            data, data_index, numeric_cols = preprocess_data(data)
            st.write("Data preprocessed successfully!")
            
            if st.sidebar.button("Create Correlation Heatmap"):
                create_correlation_heatmap(data)
            
            if st.sidebar.button("Apply PCA"):
                n_components = len(data.columns)
                pca_result = apply_pca(data, n_components)
                
                if st.sidebar.button("Plot Clusters"):
                    algorithm = AgglomerativeClustering(n_clusters=3)
                    plot_clusters(algorithm, data, pca_result, numeric_cols, data_index)

if __name__ == "__main__":
    main()
