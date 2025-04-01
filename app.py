import dash
from dash import dcc, html, Input, Output, State, callback_context
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
    # Read the CSV file
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
    
    # Create a color dictionary for consistent colors across plots
    unique_clusters = df_sample[cluster_col].astype(str).unique()
    colors = px.colors.qualitative.Bold
    color_map = {str(cluster): colors[i % len(colors)] for i, cluster in enumerate(sorted(unique_clusters))}
    
    fig_data_mds = px.scatter(
        df_mds, x='MDS1', y='MDS2', color=cluster_col,
        color_discrete_map=color_map,
        title="Data Distribution (MDS)",
        labels={'MDS1': 'Component 1', 'MDS2': 'Component 2'},
        hover_data={cluster_col: True},
        template="plotly_white"
    )
    fig_data_mds.update_layout(
        legend_title="Cluster",
        font=dict(family="Arial, sans-serif", size=13),
        title=dict(font=dict(size=16, family="Arial, sans-serif")),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    fig_data_mds.update_traces(
        marker=dict(size=9, opacity=0.8),
        hovertemplate="<b>Cluster:</b> %{marker.color}<br>" +
                     "<b>Component 1:</b> %{x:.2f}<br>" +
                     "<b>Component 2:</b> %{y:.2f}<extra></extra>"
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
        title="Variable Relationships (MDS)",
        labels={"x": "Component 1", "y": "Component 2"},
        template="plotly_white"
    )
    fig_var_mds.update_traces(
        textposition='top center',
        marker=dict(size=12, color="#3366cc", opacity=0.8, line=dict(width=1, color="#ffffff")),
        hovertemplate="<b>%{text}</b><br>" +
                     "<b>Component 1:</b> %{x:.2f}<br>" +
                     "<b>Component 2:</b> %{y:.2f}<extra></extra>"
    )
    fig_var_mds.update_layout(
        font=dict(family="Arial, sans-serif", size=13),
        title=dict(font=dict(size=16, family="Arial, sans-serif")),
        margin=dict(l=20, r=20, t=50, b=20)
    )

else:
    fig_data_mds = go.Figure()
    fig_data_mds.update_layout(
        title="Not enough valid numeric columns for Data MDS Plot",
        template="plotly_white",
        annotations=[dict(
            text="Need at least 2 numeric columns with no missing values",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            font=dict(size=14, family="Arial, sans-serif")
        )]
    )
    
    fig_var_mds = go.Figure()
    fig_var_mds.update_layout(
        title="Not enough valid numeric columns for Variable MDS Plot",
        template="plotly_white",
        annotations=[dict(
            text="Need at least 2 numeric columns with no missing values",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            font=dict(size=14, family="Arial, sans-serif")
        )]
    )

# -------------------------
# 3. Create Parallel Coordinates Plot (PCP)
# -------------------------
def create_pcp(ordering):
    """
    Generates a parallel coordinates plot using the given axes ordering.
    """
    try:
        if df.empty or len(ordering) < 1:
            fig = go.Figure()
            fig.update_layout(
                title="No data or valid ordering for PCP",
                annotations=[dict(
                    text="Please select at least one column for visualization",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=0.5
                )]
            )
            return fig
        
        # Ensure we're using only columns that exist in the dataframe
        valid_ordering = [col for col in ordering if col in df.columns]
        if not valid_ordering:
            valid_ordering = default_order[:min(8, len(default_order))]
        
        # Create a copy to avoid modifying the original dataframe
        df_pcp = df.copy()
        
        # Explicitly convert cluster column to string for consistent coloring
        df_pcp[cluster_col] = df_pcp[cluster_col].astype(str)
        
        # Create dimensions list with proper formatting
        dimensions = []
        for col in valid_ordering:
            if col in df_pcp.columns:
                if pd.api.types.is_numeric_dtype(df_pcp[col]):
                    # For numeric columns
                    dimensions.append(dict(
                        label=col,
                        values=df_pcp[col],
                        range=[df_pcp[col].min(), df_pcp[col].max()]  # Explicit range
                    ))
                else:
                    # For categorical columns
                    cat_series = df_pcp[col].astype('category')
                    codes = cat_series.cat.codes
                    categories = cat_series.cat.categories.tolist()
                    dimensions.append(dict(
                        label=col,
                        values=codes,
                        tickvals=list(range(len(categories))),
                        ticktext=categories
                    ))
        
        # Ensure we have some dimensions
        if not dimensions:
            fig = go.Figure()
            fig.update_layout(title="No valid dimensions for visualization")
            return fig
        
        # Use a fixed color scale for consistency
        colorscale = px.colors.sequential.Viridis
        
        # Create the parallel coordinates plot
        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=df_pcp[cluster_col].astype('category').cat.codes,  # Use category codes for coloring
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(title="Cluster")
            ),
            dimensions=dimensions
        ))
        
        fig.update_layout(
            title="Parallel Coordinates Visualization",
            margin=dict(l=100, r=100, t=60, b=40),
        )
        
        return fig
    except Exception as e:
        print(f"Error in create_pcp: {str(e)}")
        fig = go.Figure()
        fig.update_layout(
            title="Error creating parallel coordinates plot",
            annotations=[dict(
                text=str(e),
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.5
            )]
        )
        return fig
# -------------------------
# 4. Build the Dash App Layout
# -------------------------
# Choose a modern theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Traffic Accident Analysis"

# Custom CSS for enhanced styling
custom_css = """
body {
    font-family: 'Arial', sans-serif;
    background-color: #f7f9fc;
    color: #333;
}
.dashboard-title {
    color: #2c3e50;
    font-weight: 500;
    padding-bottom: 10px;
    border-bottom: 2px solid #3498db;
    margin-bottom: 20px;
}
.card {
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border-radius: 8px;
    border: none;
    margin-bottom: 24px;
}
.card-header {
    background-color: #f8f9fa;
    border-bottom: 1px solid #eaeaea;
    font-weight: 500;
    padding: 12px 20px;
}
.section-title {
    color: #2c3e50;
    font-weight: 500;
    margin: 20px 0 15px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #e9ecef;
}
.help-text {
    color: #6c757d;
    font-size: 0.85rem;
}
.plot-card {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    padding: 15px;
    margin-bottom: 20px;
}
.stat-card {
    background-color: white;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    text-align: center;
    margin-bottom: 15px;
    transition: transform 0.2s;
}
.stat-card:hover {
    transform: translateY(-5px);
}
.stat-value {
    font-size: 1.8rem;
    font-weight: 600;
    color: #3498db;
}
.stat-label {
    color: #7f8c8d;
    font-size: 0.85rem;
}
.control-panel {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    padding: 15px;
    margin-bottom: 20px;
}
"""

# Add custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>''' + custom_css + '''</style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if data_loaded:
    # Create summary statistics
    total_accidents = len(df)
    num_clusters = len(df[cluster_col].unique())
    largest_cluster_size = df[cluster_col].value_counts().max()
    largest_cluster_id = df[cluster_col].value_counts().idxmax()
    
    # Format the stats with commas for thousands
    total_accidents_formatted = f"{total_accidents:,}"
    largest_cluster_size_formatted = f"{largest_cluster_size:,}"
    
    app.layout = html.Div([
        # Navigation bar
        dbc.Navbar(
            dbc.Container([
                dbc.Row([
                    dbc.Col(
                        html.Img(src="https://img.icons8.com/color/48/000000/traffic-jam.png", height="40px"),
                        width="auto"
                    ),
                    dbc.Col(
                        dbc.NavbarBrand("Traffic Accident Analysis Dashboard", className="ml-2 dashboard-title"),
                    )
                ], align="center"),
                dbc.NavbarToggler(id="navbar-toggler"),
                dbc.Collapse(
                    dbc.Nav([
                        dbc.NavItem(dbc.NavLink("Data Overview", href="#overview")),
                        dbc.NavItem(dbc.NavLink("MDS Analysis", href="#mds")),
                        dbc.NavItem(dbc.NavLink("Parallel Coordinates", href="#pcp")),
                    ], className="ml-auto", navbar=True),
                    id="navbar-collapse",
                    navbar=True,
                ),
            ], fluid=True),
            color="white",
            dark=False,
            className="mb-4 shadow-sm",
        ),
        
        dbc.Container([
            # Summary statistics cards
            html.Div(id="overview"),
            html.H3("Data Overview", className="section-title"),
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        html.Div(html.I(className="fas fa-car-crash fa-2x mb-2", style={"color": "#3498db"})),
                        html.Div(total_accidents_formatted, className="stat-value"),
                        html.Div("Total Accidents", className="stat-label")
                    ], className="stat-card"),
                    width=12, sm=6, md=3
                ),
                dbc.Col(
                    dbc.Card([
                        html.Div(html.I(className="fas fa-table fa-2x mb-2", style={"color": "#2ecc71"})),
                        html.Div(str(len(numeric_cols)), className="stat-value"),
                        html.Div("Numeric Variables", className="stat-label")
                    ], className="stat-card"),
                    width=12, sm=6, md=3
                ),
                dbc.Col(
                    dbc.Card([
                        html.Div(html.I(className="fas fa-layer-group fa-2x mb-2", style={"color": "#e74c3c"})),
                        html.Div(str(num_clusters), className="stat-value"),
                        html.Div("Data Clusters", className="stat-label")
                    ], className="stat-card"),
                    width=12, sm=6, md=3
                ),
                dbc.Col(
                    dbc.Card([
                        html.Div(html.I(className="fas fa-chart-pie fa-2x mb-2", style={"color": "#9b59b6"})),
                        html.Div(largest_cluster_size_formatted, className="stat-value"),
                        html.Div(f"Largest Cluster ({largest_cluster_id})", className="stat-label")
                    ], className="stat-card"),
                    width=12, sm=6, md=3
                ),
            ], className="mb-4"),
            
            # MDS Plots section
            html.Div(id="mds"),
            html.H3("Multidimensional Scaling Analysis", className="section-title"),
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("Data Distribution (MDS)", className="mb-0"),
                            html.Small("Points represent accidents, colored by cluster", className="help-text")
                        ]),
                        dbc.CardBody(dcc.Graph(id="data-mds", figure=fig_data_mds, config={'displayModeBar': True, 'scrollZoom': True}))
                    ], className="h-100"),
                    md=6
                ),
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("Variable Relationships (MDS)", className="mb-0"),
                            html.Small("Click on variables to add them to PCP ordering", className="help-text")
                        ]),
                        dbc.CardBody(dcc.Graph(id="var-mds", figure=fig_var_mds, config={'displayModeBar': True, 'scrollZoom': True}))
                    ], className="h-100"),
                    md=6
                )
            ], className="mb-4"),
            
            # Parallel Coordinates section
            html.Div(id="pcp"),
            html.H3("Parallel Coordinates Visualization", className="section-title"),
            
            dbc.Row([
                dbc.Col(md=4, children=[
                    dbc.Card([
                        dbc.CardHeader("Visualization Controls"),
                        dbc.CardBody([
                            # First control panel
                            html.Div([
                                html.H5("Manual Ordering", className="mb-3"),
                                html.P("Select and arrange variables for PCP visualization:", className="help-text"),
                                dcc.Dropdown(
                                    id="pcp-order-dropdown",
                                    options=[{"label": col, "value": col} for col in default_order],
                                    value=default_order[:min(8, len(default_order))],  # Default to first 8 for better readability
                                    multi=True,
                                    persistence=True,
                                    persistence_type="session",
                                    className="mb-3"
                                ),
                                
                                # Tips section
                                html.Div([
                                    html.H6("Tips:", className="mt-4"),
                                    html.Ul([
                                        html.Li("Drag to reorder variables in the dropdown"),
                                        html.Li("Click and drag in the PCP to select data"),
                                        html.Li("Double-click to reset the selection")
                                    ], className="help-text pl-3")
                                ]),
                            ]),
                            
                            html.Hr(),
                            
                            # Second control panel
                            html.Div([
                                html.H5("MDS-Based Ordering", className="mb-3"),
                                html.P("Click variables in the MDS plot above to create ordering:", className="help-text"),
                                
                                html.Div(className="border p-2 bg-light mt-2 rounded", children=[
                                    html.P("Selected variables:", className="mb-1 help-text"),
                                    html.Div(id="mds-order-display", className="font-weight-bold"),
                                    dbc.Button(
                                        [html.I(className="fas fa-undo mr-2"), "Reset Selection"],
                                        id="reset-mds-order",
                                        color="secondary",
                                        size="sm",
                                        className="mt-3"
                                    )
                                ])
                            ])
                        ])
                    ], className="mb-4"),
                    
                    # Dataset info card
                    dbc.Card([
                        dbc.CardHeader("Dataset Information"),
                        dbc.CardBody([
                            dbc.ListGroup([
                                dbc.ListGroupItem([
                                    html.Strong("Data Size: "),
                                    html.Span(f"{total_accidents_formatted} rows")
                                ]),
                                dbc.ListGroupItem([
                                    html.Strong("Features: "),
                                    html.Span(f"{len(numeric_cols)} numeric, {len(cat_cols)} categorical")
                                ]),
                                dbc.ListGroupItem([
                                    html.Strong("Key Variable: "),
                                    html.Span("SeverityLevel (Low/Medium/High)")
                                ]),
                                dbc.ListGroupItem([
                                    html.Strong("Clustering: "),
                                    html.Span(f"{num_clusters} clusters with KMeans")
                                ])
                            ], flush=True)
                        ])
                    ])
                ]),
                
                dbc.Col(md=8, children=[
                    dbc.Card([
                        dbc.CardHeader("Interactive Parallel Coordinates Plot"),
                        dbc.CardBody(
                            dcc.Graph(
                                id="pcp-plot",
                                config={
                                    'displayModeBar': True, 
                                    'scrollZoom': True,
                                    'responsive': True
                                },
                                style={"height": "600px"}
                            )
                        )
                    ])
                ])
            ]),
            
            # Footer
            html.Footer([
                html.Hr(),
                html.P("Traffic Accident Analysis Dashboard • Data visualization for accident patterns and relationships",
                       className="text-center text-muted small")
            ], className="mt-5"),
            
            # Hidden components for storing data
            dcc.Store(id="mds-click-store", data=[]),
            
        ], fluid=True, className="pb-5")
    ])
else:
    app.layout = html.Div([
        dbc.Container([
            html.Div([
                html.I(className="fas fa-exclamation-triangle fa-3x text-danger mb-3"),
                html.H1("Data Loading Error", className="text-center mt-3 mb-4"),
            ], className="text-center"),
            
            dbc.Card([
                dbc.CardHeader("Error Details"),
                dbc.CardBody([
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
                ])
            ], className="shadow")
        ], className="p-5")
    ], className="d-flex align-items-center justify-content-center min-vh-100")

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
            return html.Span("No variables selected yet", className="text-muted fst-italic")
        
        # Create a nicer display with numbered badges
        items = []
        for i, var in enumerate(mds_order):
            items.append(
                dbc.Badge(
                    f"{i+1}. {var}", 
                    color="primary", 
                    className="mr-2 mb-2",
                    style={"font-size": "0.8rem"}
                )
            )
        return html.Div(items)
    
    @app.callback(
        Output("mds-click-store", "data", allow_duplicate=True),
        [Input("reset-mds-order", "n_clicks")],
        prevent_initial_call=True
    )
    def reset_mds_order(n_clicks):
        return []
    
    @app.callback(
        Output("navbar-collapse", "is_open"),
        [Input("navbar-toggler", "n_clicks")],
        [State("navbar-collapse", "is_open")],
    )
    def toggle_navbar_collapse(n, is_open):
        if n:
            return not is_open
        return is_open
    
    @app.callback(
    Output("pcp-plot", "figure"),
    [Input("pcp-order-dropdown", "value"),
     Input("mds-click-store", "data")]
)
    def update_pcp(selected_order, mds_order):
        ctx = callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
        
        # Set default ordering if nothing is selected
        if not selected_order or not isinstance(selected_order, list):
            selected_order = default_order[:min(8, len(default_order))]
        
        # If MDS variables were clicked, prioritize those
        if mds_order and len(mds_order) > 0 and trigger_id == "mds-click-store":
            # Get valid variables from MDS selection
            numerical = [col for col in mds_order if col in numeric_cols]
            # Add some categorical variables for context
            categorical = [col for col in cat_cols if col in selected_order][:3]
            new_order = numerical + categorical
        else:
            new_order = selected_order
        
        # Ensure we have a reasonable number of dimensions
        if len(new_order) > 12:
            new_order = new_order[:12]
        elif len(new_order) == 0:
            new_order = default_order[:min(8, len(default_order))]
            
        print(f"Creating PCP with variables: {new_order}")
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
                # Limit to most recent 8 selections for better readability
                current_order = current_order[-8:] if len(current_order) > 8 else current_order
        return current_order

# -------------------------
# 6. Run the App
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)