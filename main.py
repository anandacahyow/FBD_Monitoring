import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import plotly.graph_objs as go
import plotly.figure_factory as ff
from datetime import datetime, date, time

img = Image.open('Nestle_Logo.png')
st.set_page_config(page_title="DMO-P Validation Tool", page_icon=img,layout="wide")

# Step 1: Read the Excel file and preprocess the data
@st.cache
def load_data(file_path):
    df = pd.read_excel(file_path)
    # Convert 'Start Datetime' and 'End Datetime' columns to datetime
    df['Start Datetime'] = pd.to_datetime(df['Start Datetime'])
    df['End Datetime'] = pd.to_datetime(df['End Datetime'])
    return df

def format_duration(duration):
    hours = duration.seconds // 3600
    minutes = (duration.seconds % 3600) // 60
    seconds = duration.seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def create_timeline(df, default_cat, start_date, end_date, start_time, end_time, selected_categories, selected_equipment, y_axis):
    # Create a list of colors corresponding to each category
    category_colors = {
        "Production Time": "green",
        "Unplanned Stoppages": "red",
        "Not Occupied": "grey",
        "Planned Stoppages": "yellow"
    }

    # Combine start datetime with start time and end datetime with end time
    combined_start_datetime = datetime.combine(start_date, start_time)
    combined_end_datetime = datetime.combine(end_date, end_time)
    #st.write(combined_start_datetime)
    #st.write(combined_end_datetime)

    # Filter data based on selected categories and date range
    filtered_df = df[(df[default_cat].isin(selected_categories)) &
                     (df['Start Datetime'] >= combined_start_datetime) &
                     (df['End Datetime'] <= combined_end_datetime) &
                     ((df['Original Equipment'].isin(selected_equipment)) &
                      (df['Reclassified Equipment'].isin(selected_equipment)))]

    # Create a list of data for plotting
    data = []
    for index, row in filtered_df.iterrows():
        category = row['Original Category']
        start_time = row['Start Datetime']
        end_time = row['End Datetime']
        duration = end_time - start_time
        formatted_duration = format_duration(duration)
        data.append({
            'Original Equipment': row['Original Equipment'],
            'Reclassified Equipment': row['Reclassified Equipment'],
            'Category': category,
            'Original Sub Category': row['Original Sub Category'],
            'Reclassified Category': row['Reclassified Category'],
            'Reclassified Sub Category': row['Reclassified Sub Category'],
            'Start Datetime': start_time,
            'End Datetime': end_time,
            'Duration': formatted_duration,
            'PLC Code': row['PLC Code'],
            'Reclassified Reason': row['Reclassified Reason'],
            'Original Reason': row['Original Reason'],
        })

    # Create a DataFrame from the list of data
    df_plot = pd.DataFrame(data)

    if y_axis == "Original Equipment":
        colour = "Category"
        sub_cat = "Original Sub Category"
        reason = "Original Reason"
    else:
        colour = 'Reclassified Category'
        sub_cat = 'Reclassified Sub Category'
        reason = 'Reclassified Reason'
    
    # Plot the graph using Plotly Express
    fig = px.timeline(df_plot, x_start="Start Datetime", x_end="End Datetime", y=y_axis,
                      color=colour, color_discrete_map=category_colors,
                      hover_data={sub_cat: True,
                                  "Start Datetime": "|%Y-%m-%d %H:%M:%S",
                                  "End Datetime": "|%Y-%m-%d %H:%M:%S",
                                  "Duration": True,
                                  "PLC Code": True,
                                  reason: True})
    fig.update_yaxes(categoryorder="category ascending")
    fig.update_layout(title=f"🕔 Duration of {y_axis}",
                      xaxis_title="Datetime",
                      yaxis_title=y_axis,
                      width=1300,
                      height=400)
    st.plotly_chart(fig)


def create_pareto(df, category_column, value_column, duration_type, avail_cat):
    # Define category colors
    color_catalogue = {
        "Production Time": "green",
        "Unplanned Stoppages": "red",
        "Not Occupied": "grey",
        "Planned Stoppages": "yellow"
    }
    if len(df[avail_cat].unique()) == 1:
        category_colors = {}
        category_col = df[avail_cat].unique()[0]
        category_colors[category_col] = color_catalogue.get(category_col)
    else:
        category_colors = color_catalogue
        
    # Group data by category and sum the duration
    df_grouped = df.groupby(category_column)[value_column].sum().reset_index()

    # Sort categories based on the sum of duration
    df_sorted = df_grouped.sort_values(by=value_column, ascending=False)

    # Calculate cumulative percentage
    df_sorted["cumulative_percentage"] = (df_sorted[value_column].cumsum() / df_sorted[value_column].sum()) * 100

    # Plot Pareto diagram
    fig = go.Figure()

    # Add bars for frequencies with text outside the bars
    if len(df[avail_cat].unique()) == 1:
        fig.add_trace(go.Bar(
            x=df_sorted[category_column],
            y=df_sorted[value_column],
            name='Hours',
            text=df_sorted[value_column].round(2),  # Round the values to two decimal places
            textposition='outside',  # Display text outside the bars
            marker_color=list(category_colors.values())[0]
        ))
    else:
        fig.add_trace(go.Bar(
            x=df_sorted[category_column],
            y=df_sorted[value_column],
            name='Hours',
            text=df_sorted[value_column].round(2),  # Round the values to two decimal places
            textposition='outside',  # Display text outside the bars
            marker_color=[category_colors.get(category, "blue") for category in df_sorted[category_column]]  # Set bar colors based on category
        ))

    # Add the cumulative percentage line
    fig.add_trace(go.Scatter(
        x=df_sorted[category_column],
        y=df_sorted['cumulative_percentage'],
        name='Cumulative Percentage',
        line=dict(color="navy"),
        yaxis='y2'  # Secondary y-axis
    ))

    # Update the layout
    fig.update_layout(
        title=f"✅ {df[avail_cat].unique()[0] if len(df[avail_cat].unique()) == 1 else category_column} Pareto Diagram",
        height=500,
        yaxis=dict(
            title=duration_type
        ),
        yaxis2=dict(
            title='Cumulative Percentage (%)',
            overlaying='y',
            side='right'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    st.plotly_chart(fig)

def create_pareto_with_colors(df, category_column, value_column, duration_type, paretoed_param, color_column):
    # Define category colors
    color_catalogue = {
        "Production Time": "green",
        "Unplanned Stoppages": "red",
        "Not Occupied": "grey",
        "Planned Stoppages": "yellow"
    }

    # Group data by category and sum the duration
    df_grouped = df.groupby(category_column)[value_column].sum().reset_index()

    # Sort categories based on the sum of duration
    df_sorted = df_grouped.sort_values(by=value_column, ascending=False)

    # Calculate cumulative percentage
    df_sorted["cumulative_percentage"] = (df_sorted[value_column].cumsum() / df_sorted[value_column].sum()) * 100

    # Initialize list to store bar colors
    bar_colors = []

    # If color_column is provided, assign colors dynamically based on its values
    if color_column:
        for category in df_sorted[category_column]:
            corresponding_color = color_catalogue.get(df[df[category_column] == category][color_column].iloc[0], "blue")
            bar_colors.append(corresponding_color)
    else:
        # Use default color for all bars if color_column is not provided
        default_color = "blue"
        bar_colors = [default_color] * len(df_sorted)

    # Plot Pareto diagram
    fig = go.Figure()

    # Add bars for frequencies with text outside the bars
    fig.add_trace(go.Bar(
        x=df_sorted[category_column],
        y=df_sorted[value_column],
        name='Hours',
        text=df_sorted[value_column].round(2),  # Round the values to two decimal places
        textposition='outside',  # Display text outside the bars
        marker_color=bar_colors  # Set bar colors based on category
    ))

    # Add the cumulative percentage line
    fig.add_trace(go.Scatter(
        x=df_sorted[category_column],
        y=df_sorted['cumulative_percentage'],
        name='Cumulative Percentage',
        line=dict(color="navy"),
        yaxis='y2'  # Secondary y-axis
    ))

    # Update the layout
    fig.update_layout(
        title=f"📇 {paretoed_param} Pareto Diagram",
        height=500,
        yaxis=dict(
            title=duration_type
        ),
        yaxis2=dict(
            title='Cumulative Percentage (%)',
            overlaying='y',
            side='right'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    #return fig
    st.plotly_chart(fig)

def create_waterfall(df, category_column1, category_column2, value_column, duration_type):
    # Group data by category and sum the duration
    pivot_df = df.pivot_table(index=category_column1, values=value_column, aggfunc='sum')
    # Define the predefined categories
    predefined_categories = ['Not Occupied', 'Planned Stoppages', 'Production Time', 'Unplanned Stoppages']
    pivot_df = pivot_df.reindex(predefined_categories, fill_value=0)
    df_sorted1 = pivot_df.reset_index()

    # Group data by category and sum the duration
    pivot_df2 = df.pivot_table(index=category_column2, values=value_column, aggfunc='sum')
    # Define the predefined categories
    predefined_categories = ['Not Occupied', 'Planned Stoppages', 'Production Time', 'Unplanned Stoppages']
    pivot_df2 = pivot_df2.reindex(predefined_categories, fill_value=0)
    df_sorted2 = pivot_df2.reset_index()

    merged_df = pd.merge(df_sorted1, df_sorted2, left_on=category_column1, right_on=category_column2)
    merged_df.drop(columns=[category_column2], inplace=True)
    merged_df['Duration_Difference'] = merged_df['Duration_y'] - merged_df['Duration_x']
    merged_df.columns = ['Category', 'Original', 'Reclassified', 'Gap']

    #categories = list(['Ref']) + merged_df['Category'].tolist()
    #values = list([sum(merged_df['Reclassified'].tolist())]) + merged_df['Gap'].tolist()

    merged_df = merged_df.sort_values(by='Gap',ascending=False)
    categories = merged_df['Category'].tolist()
    values = merged_df['Gap'].tolist()
    values = [round(num, 2) for num in values]
    
    fig = go.Figure(go.Waterfall(
        x=categories,
        y=values,
        measure=["relative" if val != 1 else "total" for val in values],  # Different measure for each bar
        base=-10,  # Set the base to 100
        increasing=dict(marker=dict(color="green")),  # Set color for increasing values
        decreasing=dict(marker=dict(color="red")),  # Set color for decreasing values
        connector=dict(line=dict(color="grey", width=2)),  # Customize connector line
        text=values,  # Custom text for each bar
        #text=[0] + [values[i] - values[i - 1] for i in range(1, len(values))],  # Custom text for each bar
        textposition="outside",  # Set text position outside the bars
        hoverinfo="y+text",  # Display y value and custom text on hover
    ))
    # Update layout
    fig.update_layout(
        title='📈 Gap Analysis with Waterfall Graph',
        yaxis=dict(title=duration_type),
        xaxis=dict(title='Category'),
        showlegend=True,
        height=500
    )
    col1, col2 = st.columns(2)
    with col1:
        st.write("▶ Total Duration (hrs) of Original Vs Reclassification per Performance Category")
        total_sum = merged_df.sum()
        total_row = pd.DataFrame({'Category': ['Total'], 'Original': [total_sum['Original']], 'Reclassified': [total_sum['Reclassified']], 'Gap': [total_sum['Gap']]})
        merged_df = pd.concat([merged_df, total_row])
        merged_df = merged_df.reset_index(drop=True)
        st.write(merged_df)

        #reclassified_equipment = st.multiselect("Filter by Reclassified Equipment", df['Reclassified Equipment'].unique(), df['Reclassified Equipment'].unique())
        #filtered_df = df[df['Reclassified Equipment'].isin(reclassified_equipment)]
        
        # Create a pivot table
        #pivot_table = pd.pivot_table(filtered_df, values='Duration', index=['Reclassified Category', 'Original Category'], columns='Original Equipment', aggfunc='sum', fill_value=0)
        #st.write(pivot_table)
    with col2:
        st.plotly_chart(fig)
        
# Step 2: Create a Streamlit app
def main():
    st.title("📊 DMO-Performance Reclassification Validation Tools")

    # Upload file
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.sidebar.title("🔍 Data Filter:")

        # Create a multi-select dropdown for category filter in the sidebar
        default_cat = st.sidebar.selectbox("Select Category", ["Original Category", "Reclassified Category"], index=1)
        available_categories = df['Original Category'].unique()
        #selected_categories = st.sidebar.multiselect("Select categories", available_categories, default=available_categories)
        selected_categories = [category for category in available_categories if st.sidebar.checkbox(category, value=True)]

        # Create a multi-select dropdown for equipment filter in the sidebar
        available_equipment = df['Reclassified Equipment'].unique()
        #selected_equipment = st.sidebar.multiselect("Select equipment", available_equipment, default=available_equipment)
        st.sidebar.title("🛠 Choose Equipment(s):")
        all_machine_option = "All Machine"
        available_equipment_with_all = list(available_equipment)
        selected_equipment = st.sidebar.multiselect("Choose Equipment(s):", available_equipment_with_all, default=available_equipment_with_all)

        st.sidebar.title("⏳ Time Window :")
        # Create date range picker for filtering by date in the sidebar
        start_date = st.sidebar.date_input("Start Date", min_value=df['Start Datetime'].min().date(),
                                       max_value=df['End Datetime'].max().date(),
                                       value=df['Start Datetime'].min().date())
        start_time = st.sidebar.slider("Start Time", value=pd.Timestamp("06:00:00").time(), format="HH:mm:ss")
        
        end_date = st.sidebar.date_input("End Date", min_value=df['Start Datetime'].min().date(),
                                     max_value=df['End Datetime'].max().date(),
                                     value=df['End Datetime'].max().date())
        end_time = st.sidebar.slider("End Time", value=pd.Timestamp("06:00:00").time(), format="HH:mm:ss")
        
        duration_type = st.sidebar.selectbox("Select Duration units", ["Seconds", "Hours", "Days"], index=1)

        # Create bar chart with filter for Original Category
        create_timeline(df, default_cat, start_date, end_date, start_time, end_time, selected_categories, selected_equipment, "Original Equipment")

        # Create bar chart with filter for Reclassified Category
        create_timeline(df, default_cat,start_date, end_date, start_time, end_time, selected_categories, selected_equipment, "Reclassified Equipment")
                        
        combined_start_datetime = datetime.combine(start_date, start_time)
        combined_end_datetime = datetime.combine(end_date, end_time)
        
        filtered_df = df[(df[default_cat].isin(selected_categories)) &
                         (df['Start Datetime'] >= combined_start_datetime) &
                         (df['End Datetime'] <= combined_end_datetime) &
                         ((df['Original Equipment'].isin(selected_equipment)) &
                         (df['Reclassified Equipment'].isin(selected_equipment)))]

        if duration_type == 'Seconds':
            time_factor = 1
        elif duration_type == 'Hours':
            time_factor = (1/3600)
        elif duration_type == 'Days':
            time_factor = 1/(3600*24)
        filtered_df['Duration'] = time_factor*(filtered_df['End Datetime'] - filtered_df['Start Datetime']).dt.total_seconds()

        st.write("📅 DMO Event Listing")
        st.dataframe(filtered_df, height=150)
        # Create Pareto diagram for Both Category
        col1, col2 = st.columns(2)
        with col1:
            create_pareto(filtered_df, "Original Category", "Duration", duration_type, default_cat)

        with col2:
            create_pareto(filtered_df, "Reclassified Category", "Duration", duration_type, default_cat)
        
        create_waterfall(filtered_df,"Original Category","Reclassified Category", "Duration", duration_type)

        st.title("📂 Overall Line Performance (Overview)")
        header_df = filtered_df.columns.tolist()
        selected_header = st.selectbox("Choose what parameter to breakdown the Pareto:", header_df, index=header_df.index('Reclassified Reason'))

        available_category = df[default_cat].unique()
        for category in available_category:
            data_cat = filtered_df[filtered_df[default_cat] == category]
            col1, col2 = st.columns(2)
            with col1:
                create_pareto(data_cat, selected_header, "Duration", duration_type, default_cat)
            with col2:
                st.write(data_cat, height=450, width=150)


        st.title("📂 Detailed Line Performance (Specific Parameters)")
        header_df2 = filtered_df.columns.tolist()
        selected_header2 = st.selectbox("Choose what Parameter to breakdown the Pareto :", header_df2)

        header_filter = filtered_df.columns.tolist()
        selected_header_filter = st.selectbox("Choose what Parameter to be Pareto-ed:", header_filter)
        
        #filter_column = st.selectbox("Specify the :", filtered_df[selected_header_filter].unique())

        for equipment in filtered_df[selected_header_filter].unique():
            # Filter the data for the current equipment
            data_cat = filtered_df[filtered_df[selected_header_filter] == equipment]
            col1, col2 = st.columns(2)
            with col1:
                create_pareto_with_colors(data_cat, "Reclassified Reason", "Duration", duration_type, equipment, color_column='Reclassified Category')
            with col2:
                st.write(data_cat, height=450, width=150)

        
    st.sidebar.image("Nestle_Signature.png")
    st.sidebar.write("""<p style='font-size: 14px;'>This Web-App is designed to facilitate DOR member of PT Nestlé Indonesia - Panjang Factory in identifying DMO Performance Category reclassification and track compliance based on <b><a href="https://nestle.sharepoint.com/:b:/t/NMTTechnical2023/EZ2DQYyVfblDhGV11hbULU0BAPm34HHC5ZHCUERmFu3tnQ?e=IdQUp4" style="color:blue;">St-21.908-03 - Manufacturing Resources Performance Measurement Definition and Calculations</a></b></p>""", unsafe_allow_html=True)
    st.sidebar.write("""<p style='font-size: 13px;'>For any inquiries, error handling, or assistance, please feel free to reach us through Email: <br>
<a href="mailto:Ananda.Cahyo@id.nestle.com">Ananda.Cahyo@id.nestle.com <br></p>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
