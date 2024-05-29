import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import numpy as np
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
    data = pd.read_csv(file,usecols=columns)
    data['Datetime'] = pd.to_datetime(data['Date On Shift'] + ' ' + data['Time'])
    data = data[data['Shift Name'].isin(['Shift 1', 'Shift 2', 'Shift 3'])]
    data.reset_index(drop=True, inplace=True)  # Resetting index after filtering rows
    st.write(data)
    
    for i in range(1, len(data)):
        if data.loc[i, 'Datetime'] < data.loc[i - 1, 'Datetime']:
            data.loc[i, 'Datetime'] += timedelta(days=1)
            
    data = data.iloc[-8000:, :]
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

def final_dataframe(algorithms, data, pca_result, numeric_cols, data_index):
    method_names = [type(algorithm).__name__ for algorithm in algorithms]
    fig_timeseries_cluster = make_subplots(rows=len(numeric_cols), cols=1, subplot_titles=numeric_cols, shared_xaxes=True, vertical_spacing=0.01, horizontal_spacing=0.1)

    for i, algorithm in enumerate(algorithms, start=1):
        algorithm.fit(pca_result)
        
        cluster_labels = algorithm.labels_
        data.insert(0, f'Cluster_{type(algorithm).__name__}', cluster_labels)
    
        combined_data = pd.concat([data_index, data], axis=1)
    return combined_data

def plot_clusters(algorithms, data, pca_result, numeric_cols, data_index):
    method_names = [type(algorithm).__name__ for algorithm in algorithms]
    fig_timeseries_cluster = make_subplots(rows=len(numeric_cols), cols=1, subplot_titles=numeric_cols, shared_xaxes=True, vertical_spacing=0.01, horizontal_spacing=0.1)
    
    for i, algorithm in enumerate(algorithms, start=1):
        algorithm.fit(pca_result)
        
        cluster_labels = algorithm.labels_
        data.insert(0, f'Cluster_{type(algorithm).__name__}', cluster_labels)
    
        combined_data = pd.concat([data_index, data], axis=1)
        
        # Calculate centroids of each cluster
        unique_clusters = np.unique(cluster_labels)
        cluster_centroids = []
        for cluster in unique_clusters:
            cluster_data = pca_result[cluster_labels == cluster]
            centroid = np.mean(cluster_data, axis=0)
            cluster_centroids.append(centroid)
        
        # Determine leftmost, central, and rightmost clusters based on their centroids
        centroids_proj = [centroid[0] for centroid in cluster_centroids]  # Considering only the first principal component for simplicity
        leftmost_cluster = np.argmin(centroids_proj)
        rightmost_cluster = np.argmax(centroids_proj)
        central_cluster = np.argsort(centroids_proj)[len(centroids_proj)//2]
        
        # Assign colors based on centroid position
        hsl_colors = ['hsl(200, 80%, 50%)', 'hsl(100, 80%, 50%)', 'hsl(0, 80%, 50%)']
        #cluster_colors = {leftmost_cluster: 'hsl(240, 80%, 50%)', central_cluster: 'hsl(120, 80%, 50%)', rightmost_cluster: 'hsl(0, 80%, 50%)'}
        cluster_colors = {leftmost_cluster: 'hsl(0, 80%, 50%)', central_cluster: 'hsl(120, 80%, 50%)', rightmost_cluster: 'hsl(240, 80%, 50%)'}
        
        for j, col in enumerate(numeric_cols, start=1):
            cluster_col = data[f'Cluster_{type(algorithm).__name__}']
            fig_timeseries_cluster.add_trace(go.Scatter(x=data_index['Datetime'], y=data[col], mode='lines', name=col, marker=dict(color='grey', size=5)), row=j, col=1)
            fig_timeseries_cluster.add_trace(go.Scatter(x=data_index['Datetime'], y=data[col], mode='markers', name=col, marker=dict(color=[cluster_colors[c] for c in cluster_col], size=5)), row=j, col=1)
            fig_timeseries_cluster.update_xaxes(showticklabels=True, row=j, col=1)
    
    fig_timeseries_cluster.update_layout(height=8000, width=1000, title=f"MaggiPCF FBD: Measurement Timeseries Plot for {type(algorithm).__name__}", showlegend=False)
    st.plotly_chart(fig_timeseries_cluster)

def main():
    st.title("⏱ FBD-360 Clustering App")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)

        start_date = st.date_input("Start Date", value=pd.to_datetime('2024-04-15'))
        end_date = st.date_input("End Date", value=pd.to_datetime('2024-05-30'))
        
        # Create Streamlit widgets for start and end times
        start_time = st.time_input("Start Time", value=pd.to_datetime('00:00').time())
        end_time = st.time_input("End Time", value=pd.to_datetime('23:59').time())

        start_datetime = datetime.combine(start_date, start_time)
        end_datetime = datetime.combine(end_date, end_time)
        
        # Filter DataFrame based on the selected date and time ranges
        #data = data.drop_duplicates(subset=['Datetime'], keep='first')
        #data.reset_index(drop=True, inplace=True)
        st.write(start_datetime)
        st.write(end_datetime)
        st.write(data)

        
        filtered_df = data.loc[(data['Datetime'] >= start_datetime) & (data['Datetime'] <= end_datetime)]

        #filtered_data = data.copy()        
        filtered_data, data_index, numeric_cols = preprocess_data(filtered_data)
        data2, data_index2, numeric_cols2 = filtered_data.copy(), data_index.copy(), numeric_cols.copy()
        
        # Do further processing with filtered data
        n_components = len(filtered_data.columns)
        pca_result = apply_pca(filtered_data, n_components)
        
        algorithm = [AgglomerativeClustering(n_clusters=3)]
        st.write(final_dataframe(algorithm, data2, pca_result, numeric_cols2, data_index2))
        plot_clusters(algorithm, filtered_data, pca_result, numeric_cols, data_index)

if __name__ == "__main__":
    main()
