import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def load_data(file_path, columns):
    data = pd.read_csv(file_path, usecols=columns)
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
    correlation_matrix.to_excel("correlation_matrix.xlsx")
    fig_corr_heatmap = px.imshow(correlation_matrix,
                                 labels=dict(x="Numerical Columns", y="Numerical Columns"))
    fig_corr_heatmap.update_layout(title="Correlation Heatmap")
    fig_corr_heatmap.write_html("correlation_heatmap.html")

def apply_pca(data, n_components):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    
    principal_components = pca.components_
    pc_df = pd.DataFrame(principal_components, columns=data.columns)
    pc_df.to_excel("Database/PCA_components.xlsx")
    
    eigenvalues = pca.explained_variance_
    percentage = (eigenvalues / eigenvalues.sum()) * 100
    eigenvalues_df = pd.DataFrame({'Eigenvalues (Explained Variance)': eigenvalues, 'Percentage': percentage})
    eigenvalues_df.to_excel("Database/eigen.xlsx")
    
    return pca_result, principal_components

def plot_clusters(algorithms, data, pca_result, numeric_cols, data_index):
    method_names = [type(algorithm).__name__ for algorithm in algorithms]
    fig = make_subplots(rows=len(algorithms), cols=1, subplot_titles=method_names)
    fig_timeseries_cluster = make_subplots(rows=len(numeric_cols), cols=1, subplot_titles=numeric_cols, shared_xaxes=True, vertical_spacing=0.01, horizontal_spacing=0.1)

    for i, algorithm in enumerate(algorithms, start=1):
        algorithm.fit(pca_result)
        
        cluster_labels = algorithm.labels_
        data.insert(0, f'Cluster_{type(algorithm).__name__}', cluster_labels)
    
        combined_data = pd.concat([data_index, data], axis=1)
        combined_data.to_excel(f'Database/{type(algorithm).__name__}_clustered_data.xlsx', index=False)
        
        for cluster_label in np.unique(cluster_labels):
            cluster_data = pca_result[cluster_labels == cluster_label]
            sorted_clusters = sorted(
                np.unique(cluster_labels), 
                key=lambda cluster_label: np.mean(pca_result[cluster_labels == cluster_label], axis=0)[0]
            )
            text = data_index['Recipe Name']
            text2 = data_index['Datetime'].astype(str)
            fig.add_trace(go.Scatter(
                x=cluster_data[:, 0], 
                y=cluster_data[:, 1], 
                mode='markers', 
                marker=dict(color=cluster_label, colorscale='viridis', size=8), 
                name=f'Cluster_{cluster_label}',
                legendgroup=f'Cluster_{cluster_label}',
                hoverinfo='text',
                text=text + "<br>" + text2,
                showlegend=True
            ), row=i, col=1)

        fig.update_layout(title_text=type(algorithm).__name__)
        
        unique_clusters = data[f'Cluster_{type(algorithm).__name__}'].unique()
        hsl_colors = ['hsl(200, 80%, 50%)', 'hsl(100, 80%, 50%)', 'hsl(0, 80%, 50%)']
        cluster_colors = {cluster: color for cluster, color in zip(sorted_clusters, hsl_colors)}

        for j, col in enumerate(numeric_cols, start=1):
            cluster_col = data[f'Cluster_{type(algorithm).__name__}']
            fig_timeseries_cluster.add_trace(go.Scatter(x=data_index['Datetime'], y=data[col], mode='lines', name=col, marker=dict(color='grey', size=5)), row=j, col=1)
            fig_timeseries_cluster.add_trace(go.Scatter(x=data_index['Datetime'], y=data[col], mode='markers', name=col, marker=dict(color=[cluster_colors[c] for c in cluster_col], size=5)), row=j, col=1)
            fig_timeseries_cluster.update_xaxes(showticklabels=True, row=j, col=1)
    
    fig_timeseries_cluster.update_layout(height=8000, width=1000, title=f"MaggiPCF FBD: Measurement Timeseries Plot for {type(algorithm).__name__}", showlegend=False)
    fig_timeseries_cluster.show()

    fig.update_layout(title="Clustering Visualization", showlegend=True, height=3000, width=1100)
    fig.show()

def main():
    st.title('Your App Title')
    st.sidebar.title('Sidebar Title')

    # Add Streamlit widgets and organize layout here
    
    file_path = "//idpjfl0017/MAGGI/Maggi_EXT_FBDReport.csv"
    columns = ['Date On Shift', 'Time', 'Shift Name', 'Recipe Name', 'Operator Name', 'FLM  Name',
               'TR6701 Temp. 1', 'TR6702 Temp. 2', 'TR6702 Temp. 3', 'TR6702 Temp. 4', 'TR6705 Ex. Temp.',
               'PT6701 Press. 1', 'PT6702 Press. 2', 'PT6703 Press. 3', 'PT6704 Press. 4',
               'DP6701 Diff Press. 1', 'DP6702 Diff Press. 2', 'DP6703 Diff Press. 3', 'DP6704 Diff Press. 4',
               'TT6701 Mix. Temp. 1', 'TT6702 Mix. Temp. 2', 'TT6703 Mix. Temp. 3', 'TT6704 Mix. Temp. 4',
               'PT6711 Mix. Press. 1', 'PT6712 Mix. Press. 2', 'M6701 Speed Exh', 'M6702 Speed Blwr 1',
               'M6703 Speed Blwr 2', 'M6704 Speed Blwr 3', 'M6705 Speed Blwr 4', 'M6706 Speed FBD Vibr',
               'IS_SteamSupply', 'IS_SteamFlow', 'IS_SteamFlowTOT']
    
    data = load_data(file_path, columns)
    data, data_index, numeric_cols = preprocess_data(data)
    create_correlation_heatmap(data)
    
    n_components = len(data.columns)
    pca_result, principal_components = apply_pca(data, n_components)
    
    algorithms = [AgglomerativeClustering(n_clusters=3)]
    plot_clusters(algorithms, data, pca_result, numeric_cols, data_index)

if __name__ == "__main__":
    main()
