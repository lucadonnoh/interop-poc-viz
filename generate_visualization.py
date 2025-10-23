import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Read the CSV file
df = pd.read_csv('interop.csv')

# Remove rows with missing token IDs and handle NaN values
df = df.dropna(subset=['srcAbstractTokenId', 'srcValueUsd', 'duration'])
# Filter out zero or negative values for log scale
df = df[(df['srcValueUsd'] > 0) & (df['duration'] > 0)]

# Add a count column for each token type
token_counts = df['srcAbstractTokenId'].value_counts()
df['token_with_count'] = df['srcAbstractTokenId'].apply(
    lambda x: f"{x} (n={token_counts[x]})"
)

# Sort by token name for consistent colors
df = df.sort_values('srcAbstractTokenId')

# Create interactive scatter plot
fig = go.Figure()

# Get unique tokens and their counts
unique_tokens = df['srcAbstractTokenId'].unique()
colors = px.colors.qualitative.Light24

for idx, token in enumerate(unique_tokens):
    token_data = df[df['srcAbstractTokenId'] == token]
    token_count = len(token_data)

    # Dynamic opacity based on count: more points = lower opacity
    # Scale from 0.8 (few points) to 0.25 (many points)
    if token_count < 10:
        opacity = 0.8
    elif token_count < 100:
        opacity = 0.6
    elif token_count < 1000:
        opacity = 0.4
    else:
        opacity = 0.25

    fig.add_trace(go.Scattergl(  # WebGL for better performance
        x=token_data['srcValueUsd'],
        y=token_data['duration'],
        mode='markers',
        name=f"{token} (n={token_count})",
        marker=dict(
            size=6,
            opacity=opacity,
            color=colors[idx % len(colors)],
            line=dict(width=0.5, color='white')
        ),
        customdata=token_data[['transferId', 'plugin']],
        hovertemplate='<b>%{customdata[0]}</b><br>' +
                      'Plugin: %{customdata[1]}<br>' +
                      'USD Value: $%{x:.2f}<br>' +
                      'Duration: %{y}<br>' +
                      '<extra></extra>'
    ))

# Update layout for better appearance
fig.update_layout(
    title='Across Protocol Interop Transfers: USD Value vs Duration by Token<br><sub>Click legend items to show/hide tokens</sub>',
    width=1400,
    height=800,
    hovermode='closest',
    legend=dict(
        title='Token (click to toggle)',
        yanchor='top',
        y=1,
        xanchor='left',
        x=1.02,
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='gray',
        borderwidth=1
    ),
    plot_bgcolor='#f8f9fa',
    xaxis=dict(
        title='USD Value (log scale)',
        type='log',
        gridcolor='lightgray',
        showgrid=True
    ),
    yaxis=dict(
        title='Duration (log scale)',
        type='log',
        gridcolor='lightgray',
        showgrid=True
    )
)

# Save as interactive HTML
fig.write_html('index.html')
print("Interactive scatter plot saved as 'index.html'")
print("\nOpen the HTML file in your browser to interact with the chart!")
print("\nFeatures:")
print("- Click legend items to show/hide specific tokens")
print("- Double-click legend to isolate a single token")
print("- Hover over points to see detailed information")
print("- Zoom and pan by dragging on the chart")
print("- Use the toolbar in the top-right for more options")

# Print statistics
print(f"\nDataset statistics:")
print(f"Total valid transfers: {len(df)}")
print(f"Unique tokens: {df['srcAbstractTokenId'].nunique()}")
