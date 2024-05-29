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
    data = pd.read_csv(file)
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
