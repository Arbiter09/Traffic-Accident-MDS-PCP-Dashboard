import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------
# 1. Data Loading and Preparation
# -------------------------
try:
    # Read the CSV file (ensure this file is in the same folder or update the path)
    df = pd.read_csv("traffic_accidents_dict new.csv")
    
    # Drop the last two columns (if they are not relevant)
    if len(df.columns) >= 2:
        df = df.drop(df.columns[-2:], axis=1)
    
    # Drop rows with any missing values
    df = df.dropna()
    
    # Identify numeric and categorical columns from the cleaned data
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create a custom categorical column "SeverityLevel"
    # (For example, bin the first numeric column into 3 quantile-based groups)
    if numeric_cols:
        df['SeverityLevel'] = pd.qcut(df[numeric_cols[0]], q=3, labels=["Low", "Medium", "High"])
        if 'SeverityLevel' not in cat_cols:
            cat_cols.append('SeverityLevel')
    
    # -------------------------
    # Clustering: Create a cluster column using numeric data
    # -------------------------
    cluster_col = 'cluster'
    if cluster_col not in df.columns:
        print(f"'{cluster_col}' column not found. Creating clusters using KMeans...")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numeric_cols])
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)
        df[cluster_col] = clusters
    # Remove the cluster column from numeric_cols (if present)
    if cluster_col in numeric_cols:
        numeric_cols.remove(cluster_col)
    
    # For PCP default ordering, use all columns except 'cluster' and any index column.
    default_order = [col for col in df.columns if col not in [cluster_col, 'index']]
    
    data_loaded = True
    error_message = None

except Exception as e:
    import traceback
    traceback.print_exc()
    df = pd.DataFrame()
    numeric_cols = []
    cat_cols = []
    default_order = []
    cluster_col = 'cluster'
    data_loaded = False
    error_message = f"Error loading or processing data: {str(e)}"

# -------------------------
# 2. Create MDS Figures (Tasks 4a and 4b)
# -------------------------
if data_loaded and len(numeric_cols) >= 2:
    # --- Task 4(a): Data MDS Plot (Euclidean Distance) ---
    sample_size = min(1000, len(df))
    df_sample = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_sample[numeric_cols])
    
    mds_data = MDS(n_components=2, metric=True, random_state=42,
                   n_init=1, max_iter=100, eps=1e-4)
    data_mds_coords = mds_data.fit_transform(scaled_data)
    
    df_mds = pd.DataFrame({
        'MDS1': data_mds_coords[:, 0],
        'MDS2': data_mds_coords[:, 1],
        cluster_col: df_sample[cluster_col].astype(str)
    })
    
    fig_data_mds = px.scatter(
        df_mds, x='MDS1', y='MDS2', color=cluster_col,
        title="MDS Plot of Traffic Accident Data (Euclidean Distance)",
        labels={'MDS1': 'MDS Component 1', 'MDS2': 'MDS Component 2'}
    )
    fig_data_mds.update_layout(
        legend_title="Cluster ID",
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        font=dict(size=12)
    )
    
    # --- Task 4(b): Variables MDS Plot (using 1 - |correlation| distance) ---
    corr_matrix = df[numeric_cols].corr().abs()
    distance_matrix = 1 - corr_matrix
    
    mds_var = MDS(n_components=2, metric=True, dissimilarity="precomputed",
                  random_state=42, n_init=1, max_iter=100, eps=1e-4)
    var_mds_coords = mds_var.fit_transform(distance_matrix)
    
    df_var_mds = pd.DataFrame({
        'variable': numeric_cols,
        'x': var_mds_coords[:, 0],
        'y': var_mds_coords[:, 1]
    })
    
    fig_var_mds = px.scatter(
        df_var_mds, x="x", y="y", text="variable",
        title="MDS Plot of Variables (1 - |correlation| distance)",
        labels={"x": "MDS Component 1", "y": "MDS Component 2"}
    )
    fig_var_mds.update_traces(
        textposition='top center',
        marker=dict(size=12, opacity=0.8)
    )
    fig_var_mds.update_layout(
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        font=dict(size=12)
    )
    fig_var_mds.add_annotation(
        x=0.5, y=1.1,
        xref="paper", yref="paper",
        text="Click on variables to add them to PCP order",
        showarrow=False,
        font=dict(size=12, color="blue")
    )

else:
    fig_data_mds = go.Figure()
    fig_data_mds.update_layout(
        title="Not enough valid numeric columns for Data MDS Plot",
        annotations=[dict(
            text="Need at least 2 numeric columns with no missing values",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=0.5
        )]
    )
    
    fig_var_mds = go.Figure()
    fig_var_mds.update_layout(
        title="Not enough valid numeric columns for Variable MDS Plot",
        annotations=[dict(
            text="Need at least 2 numeric columns with no missing values",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=0.5
        )]
    )

# -------------------------
# 3. Create Parallel Coordinates Plot (PCP)
# -------------------------
def create_pcp(ordering):
    """
    Generates a parallel coordinates plot using the given axes ordering.
    Numeric dimensions are plotted directly.
    For categorical dimensions, we convert them to category codes and provide tick labels.
    """
    try:
        if df.empty or len(ordering) < 1:
            fig = go.Figure()
            fig.update_layout(title="No data or no valid ordering for PCP")
            return fig
        
        df_pcp = df.copy()
        valid_ordering = [col for col in ordering if col in df_pcp.columns]
        if not valid_ordering:
            valid_ordering = default_order
        
        dimensions = []
        for col in valid_ordering:
            if pd.api.types.is_numeric_dtype(df_pcp[col]):
                dimensions.append(dict(
                    label=col,
                    values=df_pcp[col]
                ))
            else:
                # Convert categorical column to category codes
                cat_series = df_pcp[col].astype('category')
                codes = cat_series.cat.codes
                categories = cat_series.cat.categories.tolist()
                dimensions.append(dict(
                    label=col,
                    values=codes,
                    tickvals=list(range(len(categories))),
                    ticktext=categories
                ))
        
        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=df_pcp[cluster_col],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Cluster")
            ),
            dimensions=dimensions
        ))
        fig.update_layout(
            title="Parallel Coordinates Plot for Traffic Accident Data",
            font=dict(size=12),
            margin=dict(l=60, r=60, t=60, b=60)
        )
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating parallel coordinates plot: {str(e)}",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=0.5
        )
        return fig

# -------------------------
# 4. Build the Dash App Layout
# -------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

if data_loaded:
    app.layout = html.Div([
        dbc.Container([
            html.H1("Traffic Accident Data Analysis Dashboard",
                    className="text-center mt-3 mb-4"),
            
            html.H2("Task 4: MDS Plots", className="mt-4"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="data-mds", figure=fig_data_mds), md=6),
                dbc.Col(dcc.Graph(id="var-mds", figure=fig_var_mds), md=6)
            ]),
            
            html.H2("Task 5 & 6: Parallel Coordinates Plot", className="mt-4"),
            
            dbc.Card([
                dbc.CardHeader("PCP Axes Ordering"),
                dbc.CardBody([
                    html.H5("Method 1: Manual Selection"),
                    html.P("Select and drag to reorder PCP axes:"),
                    dcc.Dropdown(
                        id="pcp-order-dropdown",
                        options=[{"label": col, "value": col} for col in default_order],
                        value=default_order,  # default ordering includes numeric and our custom categorical
                        multi=True,
                        persistence=True,
                        persistence_type="session"
                    ),
                    
                    html.H5("Method 2: Interactive Ordering from Variable MDS", className="mt-3"),
                    html.P("Click on variables in the Variable MDS plot to define an axis order."),
                    html.Div([
                        html.P("Current MDS-based ordering:"),
                        html.Div(id="mds-order-display", className="font-weight-bold"),
                        dbc.Button("Reset MDS Order", id="reset-mds-order",
                                   color="secondary", size="sm", className="mt-2")
                    ]),
                ])
            ], className="mb-4"),
            
            dcc.Graph(id="pcp-plot"),
            
            dcc.Store(id="mds-click-store", data=[]),
            
            dbc.Card([
                dbc.CardHeader("Dataset Information"),
                dbc.CardBody([
                    html.P(f"Total rows (after cleaning): {len(df)}"),
                    html.P(f"Numeric columns used: {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}"),
                    html.P(f"Custom categorical column: SeverityLevel"),
                    html.P(f"Cluster column: {cluster_col}"),
                ])
            ], className="mt-4")
        ], fluid=True)
    ], className="p-4")
else:
    app.layout = html.Div([
        dbc.Container([
            html.H1("Traffic Accident Dashboard - Error", className="text-center mt-3 mb-4"),
            dbc.Alert(error_message, color="danger"),
            html.P("Please check your data file and try again."),
            html.Div([
                html.H5("Troubleshooting Tips:"),
                html.Ul([
                    html.Li("Ensure 'traffic_accidents_dict new.csv' is in the same folder as this script."),
                    html.Li("Check that the CSV file is properly formatted."),
                    html.Li("Ensure you have all required libraries installed."),
                    html.Li("Confirm that you have enough data columns for MDS (at least 2 numeric columns).")
                ])
            ], className="mt-4")
        ], fluid=True)
    ], className="p-4")

# -------------------------
# 5. Dash Callbacks for Interactivity
# -------------------------
if data_loaded:
    @app.callback(
        Output("mds-order-display", "children"),
        [Input("mds-click-store", "data")]
    )
    def update_mds_order_display(mds_order):
        if not mds_order:
            return "No variables selected yet"
        return ", ".join(mds_order)
    
    @app.callback(
        Output("mds-click-store", "data", allow_duplicate=True),
        [Input("reset-mds-order", "n_clicks")],
        prevent_initial_call=True
    )
    def reset_mds_order(n_clicks):
        return []
    
    @app.callback(
        Output("pcp-plot", "figure"),
        [Input("pcp-order-dropdown", "value"),
         Input("mds-click-store", "data")]
    )
    def update_pcp(selected_order, mds_order):
        if not selected_order:
            selected_order = default_order
        
        # If the user clicked on variables in the MDS plot, use that ordering for numeric columns
        if mds_order:
            new_numerical = [col for col in mds_order if col in numeric_cols]
        else:
            new_numerical = [col for col in selected_order if col in numeric_cols]
        
        # For categorical columns, include any that were selected (like our custom "SeverityLevel")
        new_categorical = [col for col in selected_order if col in cat_cols]
        new_order = new_numerical + new_categorical
        
        return create_pcp(new_order)
    
    @app.callback(
        Output("mds-click-store", "data", allow_duplicate=True),
        [Input("var-mds", "clickData")],
        [State("mds-click-store", "data")],
        prevent_initial_call=True
    )
    def update_mds_click(clickData, current_order):
        if clickData is not None:
            clicked_variable = clickData["points"][0].get("text")
            if clicked_variable in numeric_cols:
                if clicked_variable in current_order:
                    current_order.remove(clicked_variable)
                current_order.append(clicked_variable)
                # Limit to most recent 10 selections
                current_order = current_order[-10:] if len(current_order) > 10 else current_order
        return current_order

# -------------------------
# 6. Run the App
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
