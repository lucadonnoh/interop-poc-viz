import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import plotly.io as pio
import math

# Read the data
df = pd.read_csv('interop_fresh.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Calculate statistics
total_transfers = len(df)
total_chains = df['srcChain'].nunique()
total_plugins = df['plugin'].nunique()
date_range = f"{df['timestamp'].min().strftime('%b %d %H:%M')} - {df['timestamp'].max().strftime('%b %d %H:%M')}"

print(f"Creating dashboard with {total_transfers:,} transfers...")

# ========== Create Global Token Order (for consistent colors across all charts) ==========
global_token_stats = df.dropna(subset=['srcAbstractTokenId']).groupby('srcAbstractTokenId').size().reset_index(name='count')
global_token_stats['symbol'] = global_token_stats['srcAbstractTokenId'].str.split(':').str[-1]
global_token_stats = global_token_stats.sort_values('count', ascending=False)
global_token_order = global_token_stats['symbol'].tolist()

# Define brand-accurate colors for major tokens
TOKEN_BRAND_COLORS = {
    'USDC': 'rgba(60, 145, 230, 0.8)',      # USDC Blue (brighter)
    'ETH': 'rgba(150, 100, 255, 0.8)',      # Ethereum Purple (more purple, less blue)
    'USDT': 'rgba(50, 175, 135, 0.8)',      # Tether Green (slightly brighter)
    'USDT0': 'rgba(70, 200, 160, 0.8)',     # Tether Green lighter variant (brighter)
    'WBTC': 'rgba(247, 147, 26, 0.8)',      # Bitcoin Orange
    'DAI': 'rgba(255, 184, 77, 0.8)',       # DAI Gold/Yellow
    'ZRO': 'rgba(170, 140, 255, 0.8)',      # LayerZero Purple (brighter)
    'SNX': 'rgba(0, 209, 255, 0.8)',        # Synthetix Cyan
    'POOL': 'rgba(100, 70, 160, 0.8)',      # PoolTogether Purple (brighter)
    'ACX': 'rgba(255, 88, 88, 0.8)',        # Across Red
    'WLD': 'rgba(80, 80, 80, 0.8)',         # Worldcoin Gray (brighter)
    'VLR': 'rgba(120, 220, 180, 0.8)',      # Generic teal/cyan-green
}

# Fallback palette for tokens not in the brand colors dict
token_color_palette = px.colors.qualitative.Set3

def get_token_color(token_symbol, index_fallback):
    """Get color for a token - use brand color if available, otherwise use palette"""
    if token_symbol in TOKEN_BRAND_COLORS:
        return TOKEN_BRAND_COLORS[token_symbol]
    else:
        return token_color_palette[index_fallback % len(token_color_palette)]

# Dark theme colors
DARK_BG = '#0a0e27'
CARD_BG = '#111836'
TEXT_COLOR = '#ffffff'
ACCENT_COLOR = '#00d9ff'
GRID_COLOR = '#1a2332'
MUTED_TEXT = '#a0aec0'

# Font sizes
FONT_SIZE_BASE = 12
FONT_SIZE_AXIS = 10
FONT_SIZE_LEGEND = 10
FONT_SIZE_ANNOTATION = 9
FONT_SIZE_TITLE = 14

# Protocol/Plugin colors (consistent across all charts)
PLUGIN_COLORS = {
    'relay-simple': 'rgba(99, 110, 250, 0.5)',
    'relay': 'rgba(239, 85, 59, 0.5)',
    'across': 'rgba(0, 204, 150, 0.5)',
    'stargate': 'rgba(171, 99, 250, 0.5)',
    'debridge-dln': 'rgba(255, 161, 90, 0.5)',
    'opstack-standardbridge': 'rgba(255, 255, 0, 0.6)',
    'layerzero-v2-ofts': 'rgba(25, 211, 243, 0.5)',
    'hyperlane-hwr': 'rgba(255, 102, 146, 0.5)',
    'oneinch-fusion-plus': 'rgba(50, 168, 82, 0.5)',
    'squid-coral': 'rgba(255, 105, 180, 0.5)',
    'axelar': 'rgba(139, 69, 19, 0.5)',
    'hyperlane-eco': 'rgba(255, 192, 203, 0.5)',
    'hyperlane-merkly-tokenbridge': 'rgba(147, 112, 219, 0.5)',
    'circle-gateway': 'rgba(0, 191, 255, 0.5)',
    'celer': 'rgba(255, 215, 0, 0.5)',
}

# Size bucket colors (for protocol size specialization)
SIZE_COLORS = {
    'Under $100': 'rgba(99, 110, 250, 0.8)',
    '$100-$1K': 'rgba(0, 204, 150, 0.8)',
    '$1K-$10K': 'rgba(254, 203, 82, 0.8)',
    '$10K-$100K': 'rgba(255, 161, 90, 0.8)',
    'Over $100K': 'rgba(239, 85, 59, 0.8)'
}

# Semantic colors
COLOR_POSITIVE = 'rgba(0, 204, 150, 0.8)'  # Green
COLOR_NEGATIVE = 'rgba(239, 85, 59, 0.8)'   # Red
COLOR_NEUTRAL = ACCENT_COLOR                 # Cyan
COLOR_WARNING = 'rgba(255, 161, 90, 0.8)'   # Orange
COLOR_INFO = 'rgba(99, 110, 250, 0.8)'      # Blue

# ========== 1. SANKEY DIAGRAMS ==========
print("Creating Sankey diagrams...")
df_flow = df.dropna(subset=['srcChain', 'dstChain'])
all_chains = sorted(list(set(df_flow['srcChain'].unique()) | set(df_flow['dstChain'].unique())))
chain_to_idx = {chain: idx for idx, chain in enumerate(all_chains)}
num_chains = len(all_chains)

# Sankey by transfer count
flow_data_count = df_flow.groupby(['srcChain', 'dstChain', 'plugin']).size().reset_index(name='count')
sources_count, targets_count, values_count, colors_count, labels_count = [], [], [], [], []
for _, row in flow_data_count.iterrows():
    sources_count.append(chain_to_idx[row['srcChain']])  # Left side (0-3)
    targets_count.append(chain_to_idx[row['dstChain']] + num_chains)  # Right side (4-7)
    values_count.append(row['count'])
    colors_count.append(PLUGIN_COLORS.get(row['plugin'], 'rgba(128, 128, 128, 0.3)'))
    labels_count.append(row['plugin'])

# Calculate node statistics for transfer count
# Left side nodes (sources)
node_stats_count = []
for chain in all_chains:
    outgoing = df_flow[df_flow['srcChain'] == chain]
    out_count = len(outgoing)
    out_protocols = outgoing['plugin'].nunique()
    node_stats_count.append(
        f"<b>{chain} (Source)</b><br>" +
        f"Outgoing: {out_count:,} transfers<br>" +
        f"Protocols: {out_protocols}"
    )

# Right side nodes (destinations)
for chain in all_chains:
    incoming = df_flow[df_flow['dstChain'] == chain]
    in_count = len(incoming)
    in_protocols = incoming['plugin'].nunique()
    node_stats_count.append(
        f"<b>{chain} (Destination)</b><br>" +
        f"Incoming: {in_count:,} transfers<br>" +
        f"Protocols: {in_protocols}"
    )

fig_sankey_count = go.Figure(data=[go.Sankey(
    node=dict(
        pad=30,
        thickness=20,
        line=dict(color=DARK_BG, width=1),
        label=all_chains + all_chains,  # Duplicate labels for left and right
        color=[ACCENT_COLOR] * (num_chains * 2),
        customdata=node_stats_count,
        hovertemplate='%{customdata}<extra></extra>',
        x=[0.01] * num_chains + [0.99] * num_chains  # Force left and right positioning
    ),
    link=dict(
        source=sources_count,
        target=targets_count,
        value=values_count,
        color=colors_count,
        label=labels_count,
        hovertemplate='%{label}<br>%{source.label} → %{target.label}<br>Transfers: %{value:,}<extra></extra>'
    )
)])

fig_sankey_count.update_layout(
    font=dict(size=FONT_SIZE_BASE, color=TEXT_COLOR),
    height=550,
    margin=dict(l=80, r=80, t=50, b=80, autoexpand=True),
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    title=dict(text='NETWORK FLOW (Transfer Count)', font=dict(size=FONT_SIZE_TITLE, color=TEXT_COLOR), x=0.5, xanchor='center'),
    annotations=[
        dict(
            x=0.01, y=1.05, xref='paper', yref='paper',
            text='<b>Source</b>', showarrow=False,
            font=dict(size=FONT_SIZE_AXIS, color=TEXT_COLOR),
            xanchor='left', align='left'
        ),
        dict(
            x=0.99, y=1.05, xref='paper', yref='paper',
            text='<b>Destination</b>', showarrow=False,
            font=dict(size=FONT_SIZE_AXIS, color=TEXT_COLOR),
            xanchor='right', align='right'
        )
    ]
)

# Sankey by USD value
flow_data_usd = df_flow.dropna(subset=['srcValueUsd']).groupby(['srcChain', 'dstChain', 'plugin']).agg({
    'srcValueUsd': 'sum'
}).reset_index()
flow_data_usd.columns = ['srcChain', 'dstChain', 'plugin', 'usd_volume']

sources_usd, targets_usd, values_usd, colors_usd, labels_usd = [], [], [], [], []
for _, row in flow_data_usd.iterrows():
    sources_usd.append(chain_to_idx[row['srcChain']])  # Left side (0-3)
    targets_usd.append(chain_to_idx[row['dstChain']] + num_chains)  # Right side (4-7)
    values_usd.append(row['usd_volume'])
    colors_usd.append(PLUGIN_COLORS.get(row['plugin'], 'rgba(128, 128, 128, 0.3)'))
    labels_usd.append(row['plugin'])

# Calculate node statistics for USD volume
df_usd = df_flow.dropna(subset=['srcValueUsd'])
# Left side nodes (sources)
node_stats_usd = []
for chain in all_chains:
    outgoing = df_usd[df_usd['srcChain'] == chain]
    out_volume = outgoing['srcValueUsd'].sum()
    out_protocols = outgoing['plugin'].nunique()
    node_stats_usd.append(
        f"<b>{chain} (Source)</b><br>" +
        f"Outgoing: ${out_volume:,.0f}<br>" +
        f"Protocols: {out_protocols}"
    )

# Right side nodes (destinations)
for chain in all_chains:
    incoming = df_usd[df_usd['dstChain'] == chain]
    in_volume = incoming['srcValueUsd'].sum()
    in_protocols = incoming['plugin'].nunique()
    node_stats_usd.append(
        f"<b>{chain} (Destination)</b><br>" +
        f"Incoming: ${in_volume:,.0f}<br>" +
        f"Protocols: {in_protocols}"
    )

fig_sankey_usd = go.Figure(data=[go.Sankey(
    node=dict(
        pad=30,
        thickness=20,
        line=dict(color=DARK_BG, width=1),
        label=all_chains + all_chains,  # Duplicate labels for left and right
        color=[ACCENT_COLOR] * (num_chains * 2),
        customdata=node_stats_usd,
        hovertemplate='%{customdata}<extra></extra>',
        x=[0.01] * num_chains + [0.99] * num_chains  # Force left and right positioning
    ),
    link=dict(
        source=sources_usd,
        target=targets_usd,
        value=values_usd,
        color=colors_usd,
        label=labels_usd,
        hovertemplate='%{label}<br>%{source.label} → %{target.label}<br>Volume: $%{value:,.0f}<extra></extra>'
    )
)])

fig_sankey_usd.update_layout(
    font=dict(size=FONT_SIZE_BASE, color=TEXT_COLOR),
    height=550,
    margin=dict(l=80, r=80, t=50, b=80, autoexpand=True),
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    title=dict(text='NETWORK FLOW (USD Volume)', font=dict(size=FONT_SIZE_TITLE, color=TEXT_COLOR), x=0.5, xanchor='center'),
    annotations=[
        dict(
            x=0.01, y=1.05, xref='paper', yref='paper',
            text='<b>Source</b>', showarrow=False,
            font=dict(size=FONT_SIZE_AXIS, color=TEXT_COLOR),
            xanchor='left', align='left'
        ),
        dict(
            x=0.99, y=1.05, xref='paper', yref='paper',
            text='<b>Destination</b>', showarrow=False,
            font=dict(size=FONT_SIZE_AXIS, color=TEXT_COLOR),
            xanchor='right', align='right'
        )
    ]
)

print(f"  Transfer count flows: {len(flow_data_count)}")
print(f"  USD volume flows: {len(flow_data_usd)}, total: ${flow_data_usd['usd_volume'].sum()/1e6:.1f}M")

# ========== 2. PROTOCOL PERFORMANCE ==========
print("Creating protocol performance chart...")
plugin_stats = df.groupby('plugin').agg({
    'transferId': 'count',
    'duration': ['mean', lambda x: x.quantile(0.95), lambda x: x.quantile(0.99)],
    'srcValueUsd': lambda x: x.dropna().median() if len(x.dropna()) > 0 else 0
}).reset_index()
plugin_stats.columns = ['plugin', 'total_transfers', 'avg_duration_sec', 'p95_duration_sec', 'p99_duration_sec', 'median_value_usd']

# Replace 0 duration with 0.001 (1ms) for log scale display
plugin_stats['avg_duration_sec'] = plugin_stats['avg_duration_sec'].replace(0, 0.001)
plugin_stats['p95_duration_sec'] = plugin_stats['p95_duration_sec'].replace(0, 0.001)
plugin_stats['p99_duration_sec'] = plugin_stats['p99_duration_sec'].replace(0, 0.001)

fig_performance = go.Figure()
for idx, row in plugin_stats.iterrows():
    bubble_size = max(8, min(row['median_value_usd'] / 100, 50))
    # Display as <0.001s for very small durations
    duration_display = f"{row['avg_duration_sec']:.1f}s" if row['avg_duration_sec'] >= 0.01 else "<0.01s"
    p95_display = f"{row['p95_duration_sec']:.1f}s" if row['p95_duration_sec'] >= 0.01 else "<0.01s"
    p99_display = f"{row['p99_duration_sec']:.1f}s" if row['p99_duration_sec'] >= 0.01 else "<0.01s"
    # Get plugin color or use default
    plugin_color = PLUGIN_COLORS.get(row['plugin'], 'rgba(128, 128, 128, 0.5)')
    fig_performance.add_trace(go.Scatter(
        x=[row['avg_duration_sec']],
        y=[row['total_transfers']],
        mode='markers',
        name=row['plugin'],
        marker=dict(size=bubble_size, color=plugin_color, opacity=0.7, line=dict(width=1, color=TEXT_COLOR)),
        hovertemplate=f"<b>{row['plugin']}</b><br>Transfers: {row['total_transfers']:,}<br>Avg: {duration_display} | p95: {p95_display} | p99: {p99_display}<br>Median Value: ${row['median_value_usd']:,.0f}<extra></extra>"
    ))

fig_performance.update_layout(
    font=dict(size=FONT_SIZE_BASE, color=TEXT_COLOR),
    xaxis=dict(title='Avg Duration (s)', type='log', gridcolor=GRID_COLOR, color=TEXT_COLOR),
    yaxis=dict(title='Transfer Count', type='log', gridcolor=GRID_COLOR, color=TEXT_COLOR),
    showlegend=True,
    legend=dict(font=dict(size=FONT_SIZE_LEGEND), orientation='v', x=1.02, y=1),
    height=400,
    margin=dict(l=50, r=120, t=30, b=40),
    hovermode='closest',
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    title=dict(
        text='PROTOCOL PERFORMANCE<br><sub>Bubble size = Median USD value per transfer</sub>',
        font=dict(size=FONT_SIZE_TITLE, color=TEXT_COLOR),
        x=0.5,
        xanchor='center'
    )
)

# ========== 3. HEATMAP ==========
print("Creating heatmap...")
import numpy as np
route_stats = df.dropna(subset=['srcChain', 'dstChain', 'duration']).groupby(['srcChain', 'dstChain']).agg({
    'duration': 'mean',
    'transferId': 'count'
}).reset_index()
route_stats.columns = ['srcChain', 'dstChain', 'avg_duration', 'transfer_count']

duration_matrix = np.zeros((len(all_chains), len(all_chains)))
count_matrix = np.zeros((len(all_chains), len(all_chains)))

for _, row in route_stats.iterrows():
    duration_matrix[chain_to_idx[row['srcChain']]][chain_to_idx[row['dstChain']]] = row['avg_duration']
    count_matrix[chain_to_idx[row['srcChain']]][chain_to_idx[row['dstChain']]] = row['transfer_count']

print(f"  Heatmap data range: {duration_matrix.min():.1f}s - {duration_matrix.max():.1f}s")
print(f"  Chains: {all_chains}")

annotations = []
for i, src in enumerate(all_chains):
    for j, dst in enumerate(all_chains):
        count = int(count_matrix[i][j])
        if count > 0:
            text = f"{count/1000:.1f}k" if count >= 1000 else str(count)
            annotations.append(dict(x=dst, y=src, text=text, showarrow=False, font=dict(color=TEXT_COLOR, size=FONT_SIZE_AXIS)))

fig_heatmap = go.Figure(data=go.Heatmap(
    z=duration_matrix.tolist(),  # Convert numpy array to list
    x=all_chains,
    y=all_chains,
    colorscale='Viridis',
    colorbar=dict(title=dict(text='sec', side='right', font=dict(size=FONT_SIZE_AXIS, color=TEXT_COLOR)), tickfont=dict(size=FONT_SIZE_ANNOTATION, color=TEXT_COLOR)),
    hovertemplate='%{y} → %{x}<br>Duration: %{z:.1f}s<extra></extra>',
    zmin=0,
    zmax=float(duration_matrix.max())
))

fig_heatmap.update_layout(
    font=dict(size=FONT_SIZE_BASE, color=TEXT_COLOR),
    xaxis=dict(
        title='Destination',
        side='bottom',
        color=TEXT_COLOR,
        tickfont=dict(color=TEXT_COLOR),
        gridcolor=GRID_COLOR
    ),
    yaxis=dict(
        title='Source',
        color=TEXT_COLOR,
        tickfont=dict(color=TEXT_COLOR),
        gridcolor=GRID_COLOR
    ),
    annotations=annotations,
    height=350,
    margin=dict(l=60, r=60, t=30, b=40),
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    title=dict(text='ROUTE SPEED', font=dict(size=FONT_SIZE_TITLE, color=TEXT_COLOR), x=0.5, xanchor='center')
)

# ========== 3B. TOP 3 PROTOCOLS PER ROUTE ==========
print("Creating top 3 protocols per route chart...")
route_protocol_stats = df.dropna(subset=['srcChain', 'dstChain', 'duration']).groupby(['srcChain', 'dstChain', 'plugin']).agg({
    'duration': ['mean', lambda x: x.quantile(0.95), lambda x: x.quantile(0.99)],
    'transferId': 'count'
}).reset_index()
route_protocol_stats.columns = ['srcChain', 'dstChain', 'plugin', 'avg_duration', 'p95_duration', 'p99_duration', 'transfer_count']

# Get top routes by transfer count (exclude same-chain transfers)
all_routes = df.dropna(subset=['srcChain', 'dstChain']).groupby(['srcChain', 'dstChain']).size().sort_values(ascending=False)
top_routes = all_routes[all_routes.index.get_level_values(0) != all_routes.index.get_level_values(1)].head(12)

# For each route, find top 3 protocols by speed
route_labels = []
route_top3_data = {}  # Will store {route: [(plugin, duration, count), ...]}

for route, total_count in top_routes.items():
    src, dst = route
    route_label = f"{src}→{dst}"
    route_labels.append(route_label)

    # Get all protocols for this route
    route_data = route_protocol_stats[(route_protocol_stats['srcChain'] == src) &
                                      (route_protocol_stats['dstChain'] == dst)]

    if len(route_data) > 0:
        # Sort by duration and get top 3
        top3 = route_data.nsmallest(3, 'avg_duration')
        route_top3_data[route_label] = [
            (row['plugin'], row['avg_duration'], row['p95_duration'], row['p99_duration'], row['transfer_count'])
            for _, row in top3.iterrows()
        ]
    else:
        route_top3_data[route_label] = []

# Create figure with grouped bars - one trace per rank (1st, 2nd, 3rd)
fig_route_comparison = go.Figure()

for rank in range(3):
    rank_labels = ['🥇 Fastest', '🥈 2nd Fastest', '🥉 3rd Fastest']
    durations = []
    hover_texts = []
    colors_for_rank = []
    protocol_names = []

    for route_label in route_labels:
        top3 = route_top3_data[route_label]

        if len(top3) > rank:
            plugin, duration, p95, p99, transfer_count = top3[rank]
            durations.append(duration)
            colors_for_rank.append(PLUGIN_COLORS.get(plugin, 'rgba(128, 128, 128, 0.5)'))
            hover_texts.append(f"<b>{route_label}</b><br>{rank_labels[rank]}: {plugin}<br>Avg: {duration:.1f}s | p95: {p95:.1f}s | p99: {p99:.1f}s<br>Transfers: {transfer_count:,}")
            protocol_names.append(plugin)
        else:
            durations.append(None)
            colors_for_rank.append('rgba(200, 200, 200, 0.5)')
            hover_texts.append("")
            protocol_names.append("")

    fig_route_comparison.add_trace(go.Bar(
        x=route_labels,
        y=durations,
        name=rank_labels[rank],
        marker=dict(color=colors_for_rank),
        text=protocol_names,
        textposition='outside',
        textfont=dict(size=9),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_texts
    ))

fig_route_comparison.update_layout(
    font=dict(size=FONT_SIZE_BASE, color=TEXT_COLOR),
    xaxis=dict(title='Route', color=TEXT_COLOR, tickfont=dict(size=FONT_SIZE_AXIS)),
    yaxis=dict(title='Avg Duration (s)', color=TEXT_COLOR, gridcolor=GRID_COLOR),
    barmode='group',
    height=400,
    margin=dict(l=50, r=20, t=60, b=100),
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    legend=dict(font=dict(size=FONT_SIZE_LEGEND), orientation='h', y=1.12, x=0.5, xanchor='center'),
    title=dict(text='TOP 3 PROTOCOLS BY SPEED PER ROUTE', font=dict(size=FONT_SIZE_TITLE, color=TEXT_COLOR), x=0.5, xanchor='center')
)

print(f"  Top 3 protocols identified for {len(top_routes)} routes")

# ========== 4. TIMELINE - TRANSFER COUNT ==========
print("Creating timeline...")
df_time = df.set_index('timestamp')
hourly_by_plugin = df_time.groupby([pd.Grouper(freq='1h'), 'plugin']).size().reset_index(name='count')

# Filter out the last incomplete hour
last_hour = hourly_by_plugin['timestamp'].max()
hourly_by_plugin = hourly_by_plugin[hourly_by_plugin['timestamp'] < last_hour]

top_plugins = df['plugin'].value_counts().head(8).index.tolist()
df_top = hourly_by_plugin[hourly_by_plugin['plugin'].isin(top_plugins)]

fig_timeline_count = go.Figure()
for plugin in top_plugins:
    plugin_data = df_top[df_top['plugin'] == plugin]
    plugin_color = PLUGIN_COLORS.get(plugin, 'rgba(128, 128, 128, 0.5)')
    fig_timeline_count.add_trace(go.Scatter(
        x=plugin_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),  # Convert timestamp to string
        y=plugin_data['count'].astype(int).tolist(),  # Convert to int list
        mode='lines', name=plugin, stackgroup='one', line=dict(width=0), fillcolor=plugin_color,
        hovertemplate='<b>%{fullData.name}</b><br>%{y:,} transfers<extra></extra>'
    ))

print(f"  Timeline total transfers: {df_top['count'].sum()}, max/hour: {df_top.groupby('timestamp')['count'].sum().max()}")

fig_timeline_count.update_layout(
    font=dict(size=FONT_SIZE_BASE, color=TEXT_COLOR),
    xaxis=dict(title='', gridcolor=GRID_COLOR, color=TEXT_COLOR, tickfont=dict(size=FONT_SIZE_AXIS)),
    yaxis=dict(title='Transfers/hr', gridcolor=GRID_COLOR, color=TEXT_COLOR, rangemode='tozero'),
    hovermode='x unified',
    height=350,
    margin=dict(l=50, r=10, t=100, b=40),
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    showlegend=True,
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='center',
        x=0.5,
        font=dict(size=FONT_SIZE_LEGEND)
    ),
    title=dict(text='TRANSFER COUNT TIMELINE', font=dict(size=FONT_SIZE_TITLE, color=TEXT_COLOR), x=0.5, xanchor='center')
)

# ========== 4B. TIMELINE - USD VOLUME ==========
print("Creating USD volume timeline...")
df_time_usd = df.dropna(subset=['srcValueUsd']).set_index('timestamp')
hourly_by_plugin_usd = df_time_usd.groupby([pd.Grouper(freq='1h'), 'plugin'])['srcValueUsd'].sum().reset_index(name='usd_volume')

# Filter out the last incomplete hour
last_hour_usd = hourly_by_plugin_usd['timestamp'].max()
hourly_by_plugin_usd = hourly_by_plugin_usd[hourly_by_plugin_usd['timestamp'] < last_hour_usd]

df_top_usd = hourly_by_plugin_usd[hourly_by_plugin_usd['plugin'].isin(top_plugins)]

fig_timeline_usd = go.Figure()
for plugin in top_plugins:
    plugin_data = df_top_usd[df_top_usd['plugin'] == plugin]
    plugin_color = PLUGIN_COLORS.get(plugin, 'rgba(128, 128, 128, 0.5)')
    fig_timeline_usd.add_trace(go.Scatter(
        x=plugin_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        y=plugin_data['usd_volume'].tolist(),
        mode='lines', name=plugin, stackgroup='one', line=dict(width=0), fillcolor=plugin_color,
        hovertemplate='<b>%{fullData.name}</b><br>$%{y:,.0f}<extra></extra>'
    ))

total_usd_timeline = df_top_usd['usd_volume'].sum()
max_hour_usd = df_top_usd.groupby('timestamp')['usd_volume'].sum().max()
print(f"  Timeline total USD volume: ${total_usd_timeline/1e6:.1f}M, max/hour: ${max_hour_usd/1e3:.0f}K")

fig_timeline_usd.update_layout(
    font=dict(size=FONT_SIZE_BASE, color=TEXT_COLOR),
    xaxis=dict(title='', gridcolor=GRID_COLOR, color=TEXT_COLOR, tickfont=dict(size=FONT_SIZE_AXIS)),
    yaxis=dict(title='USD/hr', gridcolor=GRID_COLOR, color=TEXT_COLOR, rangemode='tozero'),
    hovermode='x unified',
    height=350,
    margin=dict(l=50, r=10, t=80, b=40),
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    showlegend=True,
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='center',
        x=0.5,
        font=dict(size=FONT_SIZE_LEGEND)
    ),
    title=dict(text='USD VOLUME TIMELINE', font=dict(size=FONT_SIZE_TITLE, color=TEXT_COLOR), x=0.5, xanchor='center')
)

# ========== 5. NET FLOWS ==========
print("Creating net flows chart...")
df_with_usd = df.dropna(subset=['srcValueUsd'])

# Calculate net flows for each chain (incoming - outgoing)
net_flows = {}
for chain in all_chains:
    outgoing = df_with_usd[df_with_usd['srcChain'] == chain]['srcValueUsd'].sum()
    incoming = df_with_usd[df_with_usd['dstChain'] == chain]['srcValueUsd'].sum()
    net_flows[chain] = incoming - outgoing

# Sort by net flow
net_flows_sorted = dict(sorted(net_flows.items(), key=lambda x: x[1]))
chains_ordered = list(net_flows_sorted.keys())
net_values = list(net_flows_sorted.values())

# Color based on positive/negative
colors_net = [COLOR_POSITIVE if v >= 0 else COLOR_NEGATIVE for v in net_values]

# Format text labels to be shorter
text_labels = []
for v in net_values:
    if abs(v) >= 1e6:
        text_labels.append(f"${v/1e6:.2f}M")
    elif abs(v) >= 1e3:
        text_labels.append(f"${v/1e3:.0f}K")
    else:
        text_labels.append(f"${v:.0f}")

# Calculate appropriate x-axis range
max_val = max(abs(v) for v in net_values)
x_range = [-max_val * 1.15, max_val * 1.15]  # 15% padding for text labels

fig_net_flows = go.Figure(data=[go.Bar(
    y=chains_ordered,
    x=net_values,
    orientation='h',
    marker=dict(color=colors_net),
    hovertemplate='<b>%{y}</b><br>Net Flow: $%{x:,.0f}<extra></extra>',
    text=text_labels,
    textposition='outside',
    textfont=dict(size=FONT_SIZE_AXIS, color=TEXT_COLOR),
    cliponaxis=False
)])

fig_net_flows.update_layout(
    font=dict(size=FONT_SIZE_BASE, color=TEXT_COLOR),
    xaxis=dict(
        title='Net USD Flow (Incoming - Outgoing)',
        gridcolor=GRID_COLOR,
        color=TEXT_COLOR,
        zeroline=True,
        zerolinecolor='rgba(255, 255, 255, 0.3)',
        zerolinewidth=2,
        fixedrange=False,
        range=x_range
    ),
    yaxis=dict(
        title='',
        color=TEXT_COLOR,
        tickfont=dict(size=FONT_SIZE_AXIS),
        ticksuffix="        "  # Add spacing after y-axis labels to prevent overlap with bar text
    ),
    height=400,
    margin=dict(l=100, r=20, t=30, b=50),
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    showlegend=False,
    title=dict(text='NET USD FLOWS', font=dict(size=FONT_SIZE_TITLE, color=TEXT_COLOR), x=0.5, xanchor='center'),
    bargap=0.3,
    uniformtext_minsize=FONT_SIZE_ANNOTATION,
    uniformtext_mode='show'
)

print(f"  Net flows: {', '.join([f'{k}: ${v:,.0f}' for k, v in net_flows_sorted.items()])}")

# ========== 6. PROTOCOL TRANSFER SIZE SPECIALIZATION ==========
print("Creating protocol size specialization chart...")
df_with_size = df.dropna(subset=['srcValueUsd']).copy()

# Categorize transfer sizes
def categorize_size(usd):
    if usd < 100:
        return 'Under $100'
    elif usd < 1000:
        return '$100-$1K'
    elif usd < 10000:
        return '$1K-$10K'
    elif usd < 100000:
        return '$10K-$100K'
    else:
        return 'Over $100K'

df_with_size['size_bucket'] = df_with_size['srcValueUsd'].apply(categorize_size)

# Get top 10 protocols by transfer count
top_protocols_for_size = df_with_size['plugin'].value_counts().head(10).index.tolist()
df_size_filtered = df_with_size[df_with_size['plugin'].isin(top_protocols_for_size)]

# Calculate distribution for each protocol
size_order = ['Under $100', '$100-$1K', '$1K-$10K', '$10K-$100K', 'Over $100K']

protocol_size_dist = df_size_filtered.groupby(['plugin', 'size_bucket']).size().unstack(fill_value=0)
# Calculate percentages
protocol_size_pct = protocol_size_dist.div(protocol_size_dist.sum(axis=1), axis=0) * 100
# Sort by average transfer size (weighted)
size_weights = {'Under $100': 50, '$100-$1K': 500, '$1K-$10K': 5000, '$10K-$100K': 50000, 'Over $100K': 200000}
protocol_size_pct['weighted_avg'] = sum(protocol_size_pct.get(size, 0) * size_weights[size] for size in size_order if size in protocol_size_pct.columns)
protocol_size_pct = protocol_size_pct.sort_values('weighted_avg')
protocol_size_pct = protocol_size_pct.drop('weighted_avg', axis=1)

fig_protocol_size = go.Figure()
# Assign legendrank to control legend order (higher rank = later in legend)
legend_ranks = {
    'Under $100': 5,
    '$100-$1K': 4,
    '$1K-$10K': 3,
    '$10K-$100K': 2,
    'Over $100K': 1
}
for size in size_order:
    if size in protocol_size_pct.columns:
        # Get raw counts for tooltip
        raw_counts = [protocol_size_dist.loc[protocol, size] if size in protocol_size_dist.columns else 0
                     for protocol in protocol_size_pct.index]

        fig_protocol_size.add_trace(go.Bar(
            name=size,
            y=protocol_size_pct.index.tolist(),
            x=protocol_size_pct[size].tolist(),
            orientation='h',
            marker=dict(color=SIZE_COLORS[size]),
            customdata=raw_counts,
            hovertemplate='<b>%{y}</b><br>' + size + ': %{x:.1f}%<br>Count: %{customdata:,} transfers<extra></extra>',
            legendrank=legend_ranks[size]
        ))

fig_protocol_size.update_layout(
    barmode='stack',
    font=dict(size=FONT_SIZE_BASE, color=TEXT_COLOR),
    xaxis=dict(
        title='% of Transfers',
        gridcolor=GRID_COLOR,
        color=TEXT_COLOR,
        ticksuffix='%'
    ),
    yaxis=dict(
        title='',
        color=TEXT_COLOR,
        tickfont=dict(size=FONT_SIZE_AXIS),
        automargin=True,
        ticksuffix="   "
    ),
    height=350,
    margin=dict(l=120, r=20, t=80, b=50, pad=10),
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    legend=dict(
        title='Transfer Size',
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='center',
        x=0.5,
        font=dict(size=FONT_SIZE_LEGEND)
    ),
    title=dict(text='PROTOCOL TRANSFER SIZE SPECIALIZATION', font=dict(size=FONT_SIZE_TITLE, color=TEXT_COLOR), x=0.5, xanchor='center', y=0.98, yanchor='top')
)

print(f"  Size specialization: {len(top_protocols_for_size)} protocols analyzed")

# ========== 7. TOKEN MARKET OVERVIEW ==========
print("Creating token market overview...")
token_stats = df_with_size.groupby('srcAbstractTokenId').agg({
    'transferId': 'count',
    'srcValueUsd': ['sum', 'median']
}).reset_index()
token_stats.columns = ['token', 'transfer_count', 'total_usd', 'median_usd']

# Extract token symbol from MOCK01:circle:USDC format
token_stats['symbol'] = token_stats['token'].str.split(':').str[-1]

# Filter out tokens with zero or negligible volume (less than $1)
token_stats = token_stats[token_stats['total_usd'] >= 1]

# Get top 15 tokens by transfer count
top_tokens = token_stats.nlargest(15, 'transfer_count')

fig_tokens = go.Figure()

for _, row in top_tokens.iterrows():
    median_value = row['median_usd']
    # Scale by median transfer value (log scale for better distribution)
    bubble_size = min(50, max(10, 15 + math.log10(max(1, median_value)) * 5))

    # Assign color based on brand colors or global token order
    token_symbol = row['symbol']
    if token_symbol in global_token_order:
        token_color = get_token_color(token_symbol, global_token_order.index(token_symbol))
    else:
        token_color = 'rgba(128, 128, 128, 0.7)'  # Gray for unknown tokens

    fig_tokens.add_trace(go.Scatter(
        x=[row['transfer_count']],
        y=[row['total_usd']],
        mode='markers+text',
        name=row['symbol'],
        marker=dict(
            size=bubble_size,
            color=token_color,
            opacity=0.7,
            line=dict(width=2, color=TEXT_COLOR)
        ),
        text=[row['symbol']],
        textposition='top center',
        textfont=dict(size=FONT_SIZE_AXIS, color=TEXT_COLOR),
        hovertemplate=(
            f"<b>{row['symbol']}</b><br>" +
            f"Transfers: {row['transfer_count']:,}<br>" +
            f"Volume: ${row['total_usd']:,.0f}<br>" +
            f"Median: ${row['median_usd']:,.0f}<br>" +
            "<extra></extra>"
        )
    ))

fig_tokens.update_layout(
    font=dict(size=FONT_SIZE_BASE, color=TEXT_COLOR),
    xaxis=dict(
        title='Transfer Count',
        gridcolor=GRID_COLOR,
        color=TEXT_COLOR,
        type='log'
    ),
    yaxis=dict(
        title='Total USD Volume',
        gridcolor=GRID_COLOR,
        color=TEXT_COLOR,
        type='log'
    ),
    showlegend=False,
    height=350,
    margin=dict(l=50, r=20, t=30, b=50),
    hovermode='closest',
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    title=dict(
        text='TOKEN MARKET OVERVIEW<br><sub>Bubble size = Median transfer value</sub>',
        font=dict(size=FONT_SIZE_TITLE, color=TEXT_COLOR),
        x=0.5,
        xanchor='center'
    )
)

if len(top_tokens) > 0:
    print(f"  Token overview: {len(top_tokens)} tokens, {top_tokens.iloc[0]['symbol']} leads with {top_tokens.iloc[0]['transfer_count']:,} transfers")
else:
    print(f"  Token overview: No tokens with valid srcSymbol data")

# ========== 7B. PER-CHAIN TOKEN FLOW ==========
print("Creating per-chain token flow...")
chain_token_stats = df.dropna(subset=['srcChain', 'srcAbstractTokenId']).copy()
chain_token_stats['symbol'] = chain_token_stats['srcAbstractTokenId'].str.split(':').str[-1]
chain_token_counts = chain_token_stats.groupby(['srcChain', 'symbol']).size().reset_index(name='count')

# Get top 6 tokens overall (use global token order)
top_token_symbols = [token for token in global_token_order if token in chain_token_stats['symbol'].values][:6]

fig_chain_tokens = go.Figure()

for token in top_token_symbols:
    token_data = []
    for chain in all_chains:
        count = chain_token_counts[(chain_token_counts['srcChain'] == chain) & (chain_token_counts['symbol'] == token)]['count'].sum()
        token_data.append(count)

    # Assign color based on brand colors or global token order
    token_color = get_token_color(token, global_token_order.index(token))

    fig_chain_tokens.add_trace(go.Bar(
        name=token,
        x=all_chains,
        y=token_data,
        marker=dict(color=token_color),
        hovertemplate=f'<b>{token}</b><br>%{{x}}: %{{y:,}} transfers<extra></extra>'
    ))

fig_chain_tokens.update_layout(
    font=dict(size=FONT_SIZE_BASE, color=TEXT_COLOR),
    xaxis=dict(title='Chain', color=TEXT_COLOR, tickfont=dict(size=FONT_SIZE_AXIS)),
    yaxis=dict(title='Outgoing Transfers (Top 6 Tokens)', color=TEXT_COLOR, gridcolor=GRID_COLOR),
    barmode='stack',
    height=350,
    margin=dict(l=50, r=20, t=60, b=60),
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    legend=dict(font=dict(size=FONT_SIZE_LEGEND), orientation='h', y=1.15, x=0.5, xanchor='center'),
    title=dict(text='PER-CHAIN TOKEN FLOW', font=dict(size=FONT_SIZE_TITLE, color=TEXT_COLOR), x=0.5, xanchor='center')
)

print(f"  Chain token flow: {len(top_token_symbols)} tokens across {len(all_chains)} chains")

# ========== 8. MISSING FINANCIAL DATA ==========
print("Creating missing data chart...")
df_missing = df[df['srcAbstractTokenId'].isna()].copy()

# Top protocols with missing data - calculate percentages
missing_by_protocol = df_missing.groupby('plugin').size().reset_index(name='missing_transfers')

# Calculate total transfers per protocol
total_by_protocol = df.groupby('plugin').size().reset_index(name='total_transfers')

# Merge to get both missing and total
missing_by_protocol = missing_by_protocol.merge(total_by_protocol, on='plugin', how='left')

# Calculate percentage
missing_by_protocol['missing_pct'] = (missing_by_protocol['missing_transfers'] / missing_by_protocol['total_transfers']) * 100

# Sort by percentage (highest first), then take top 10
missing_by_protocol = missing_by_protocol.sort_values('missing_pct', ascending=False).head(10)
missing_by_protocol = missing_by_protocol.sort_values('missing_pct', ascending=True)  # Sort for display (low to high)

# Top token addresses needing abstractTokenId mapping (exclude native ETH and predeploys)
missing_tokens_raw = df_missing.groupby(['srcTokenAddress', 'srcChain']).size().reset_index(name='transfers')

# Strip padding from addresses and filter out native tokens
def strip_padding(addr):
    if pd.isna(addr) or addr == 'native':
        return addr
    # Strip leading zeros padding: 0x000000000000000000000000 -> 0x
    if addr.startswith('0x000000000000000000000000'):
        return '0x' + addr[26:]
    return addr

missing_tokens_raw['srcTokenAddress'] = missing_tokens_raw['srcTokenAddress'].apply(strip_padding)

# Filter out native tokens and common predeploys
predeploys = [
    'native',
    '0x0000000000000000000000000000000000000000',  # Zero address
    '0x4200000000000000000000000000000000000006',  # WETH predeploy on OP chains
]

missing_tokens_raw = missing_tokens_raw[
    ~missing_tokens_raw['srcTokenAddress'].isin(predeploys) &
    missing_tokens_raw['srcTokenAddress'].notna()
]

# Aggregate by stripped address and chain
missing_tokens_raw = missing_tokens_raw.groupby(['srcTokenAddress', 'srcChain'])['transfers'].sum().reset_index()

missing_tokens_raw = missing_tokens_raw.sort_values('transfers', ascending=False).head(15)

if len(missing_tokens_raw) > 0:
    missing_tokens_raw = missing_tokens_raw.sort_values('transfers', ascending=True)  # Sort for display (low to high)

    # Fetch token symbols from chain with caching
    from web3 import Web3
    import os

    rpc_endpoints = {
        'ethereum': 'https://eth.llamarpc.com',
        'arbitrum': 'https://arb1.arbitrum.io/rpc',
        'optimism': 'https://mainnet.optimism.io',
        'base': 'https://mainnet.base.org'
    }

    # Load symbol cache
    cache_file = 'token_symbol_cache.json'
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            symbol_cache = json.load(f)
    else:
        symbol_cache = {}

    # ERC20 symbol() function signature
    symbol_abi = [{
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function"
    }]

    def fetch_token_symbol(address, chain):
        cache_key = f"{chain}:{address.lower()}"

        # Check cache first
        if cache_key in symbol_cache:
            return symbol_cache[cache_key]

        # Fetch from chain
        try:
            if chain not in rpc_endpoints:
                return '?'

            w3 = Web3(Web3.HTTPProvider(rpc_endpoints[chain]))
            contract = w3.eth.contract(address=Web3.to_checksum_address(address), abi=symbol_abi)
            symbol = contract.functions.symbol().call()

            # Cache the result
            symbol_cache[cache_key] = symbol
            return symbol
        except Exception as e:
            symbol_cache[cache_key] = '?'
            return '?'

    print("  Fetching token symbols from blockchain...")
    missing_tokens_raw['symbol'] = missing_tokens_raw.apply(
        lambda row: fetch_token_symbol(row['srcTokenAddress'], row['srcChain']),
        axis=1
    )

    # Save updated cache
    with open(cache_file, 'w') as f:
        json.dump(symbol_cache, f, indent=2)
    print(f"  Cached {len(symbol_cache)} token symbols")

    # Create readable labels with fetched symbols
    def format_token_label(row):
        addr = row['srcTokenAddress'].lower()
        chain = row['srcChain']
        symbol = row['symbol']

        if len(addr) > 20:
            return f"{addr[:6]}...{addr[-4:]} [{symbol}] ({chain})"
        else:
            return f"{addr} [{symbol}] ({chain})"

    missing_tokens = missing_tokens_raw.copy()
    missing_tokens['display'] = missing_tokens.apply(format_token_label, axis=1)
else:
    # Create empty dataframe if no tokens remain after filtering
    missing_tokens = pd.DataFrame({'display': ['No unmapped tokens'], 'transfers': [0]})

# Create two subplots side by side
from plotly.subplots import make_subplots

fig_missing = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Top Protocols Missing Abstract Token IDs', 'Token Addresses Needing Abstract Token ID Mapping'),
    horizontal_spacing=0.15
)

# Left: Protocols
fig_missing.add_trace(
    go.Bar(
        y=missing_by_protocol['plugin'].tolist(),
        x=missing_by_protocol['missing_pct'].tolist(),
        orientation='h',
        marker=dict(color=COLOR_NEGATIVE),
        customdata=missing_by_protocol['missing_transfers'].tolist(),
        hovertemplate='<b>%{y}</b><br>Missing: %{x:.1f}%<br>Count: %{customdata:,} transfers<extra></extra>',
        showlegend=False
    ),
    row=1, col=1
)

# Right: Token addresses
fig_missing.add_trace(
    go.Bar(
        y=missing_tokens['display'].tolist(),
        x=missing_tokens['transfers'].tolist(),
        orientation='h',
        marker=dict(color=COLOR_WARNING),
        hovertemplate='<b>%{y}</b><br>Missing: %{x:,} transfers<extra></extra>',
        showlegend=False
    ),
    row=1, col=2
)

fig_missing.update_xaxes(title_text="% of Protocol's Transfers Missing Abstract Token ID", row=1, col=1, gridcolor=GRID_COLOR, color=TEXT_COLOR)
fig_missing.update_xaxes(title_text="Transfers Needing Mapping", row=1, col=2, gridcolor=GRID_COLOR, color=TEXT_COLOR)
fig_missing.update_yaxes(tickfont=dict(size=FONT_SIZE_AXIS), color=TEXT_COLOR, automargin=True, row=1, col=1)
fig_missing.update_yaxes(tickfont=dict(size=FONT_SIZE_ANNOTATION), color=TEXT_COLOR, automargin=True, row=1, col=2)

# Overall stats for annotation
total_transfers = len(df)
missing_transfers = len(df_missing)
coverage_pct = 100 * (1 - missing_transfers / total_transfers)

fig_missing.update_layout(
    font=dict(size=FONT_SIZE_BASE, color=TEXT_COLOR),
    height=400,
    margin=dict(l=10, r=10, t=80, b=50, pad=10),
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    title=dict(
        text=f'MISSING ABSTRACT TOKEN IDS: {missing_transfers:,} transfers ({100-coverage_pct:.1f}% of total)<br><sub>Map these token addresses to abstractTokenIds to enable financial tracking</sub>',
        font=dict(size=FONT_SIZE_TITLE, color=TEXT_COLOR),
        x=0.5,
        xanchor='center'
    )
)

print(f"  Missing data: {missing_transfers:,} transfers ({100-coverage_pct:.1f}%), top issues by %: {missing_by_protocol.iloc[-1]['plugin']} ({missing_by_protocol.iloc[-1]['missing_pct']:.1f}%), {missing_by_protocol.iloc[-2]['plugin']} ({missing_by_protocol.iloc[-2]['missing_pct']:.1f}%)")

# ========== 9. TRANSFER VALUE DISTRIBUTION ==========
print("Creating transfer value distribution...")
df_with_usd_values = df.dropna(subset=['srcValueUsd']).copy()

# Create categorical buckets
def categorize_value(val):
    if val < 10:
        return '<$10'
    elif val < 50:
        return '$10-$50'
    elif val < 100:
        return '$50-$100'
    elif val < 500:
        return '$100-$500'
    elif val < 1000:
        return '$500-$1K'
    elif val < 5000:
        return '$1K-$5K'
    elif val < 10000:
        return '$5K-$10K'
    elif val < 50000:
        return '$10K-$50K'
    elif val < 100000:
        return '$50K-$100K'
    else:
        return '>$100K'

df_with_usd_values['bucket'] = df_with_usd_values['srcValueUsd'].apply(categorize_value)

# Define bucket order
bucket_order = ['<$10', '$10-$50', '$50-$100', '$100-$500', '$500-$1K',
                '$1K-$5K', '$5K-$10K', '$10K-$50K', '$50K-$100K', '>$100K']

# Count transfers and sum volume per bucket
bucket_stats = df_with_usd_values.groupby('bucket')['srcValueUsd'].agg(['count', 'sum'])
bucket_counts = [bucket_stats.loc[bucket, 'count'] if bucket in bucket_stats.index else 0 for bucket in bucket_order]
bucket_volumes = [bucket_stats.loc[bucket, 'sum'] if bucket in bucket_stats.index else 0 for bucket in bucket_order]

# Create side-by-side charts
from plotly.subplots import make_subplots

fig_distribution = make_subplots(
    rows=1, cols=2,
    horizontal_spacing=0.12
)

# Left: Transfer count
fig_distribution.add_trace(
    go.Bar(
        x=bucket_order,
        y=bucket_counts,
        marker=dict(color=COLOR_INFO),
        hovertemplate='%{x}<br>Transfers: %{y:,}<extra></extra>',
        showlegend=False
    ),
    row=1, col=1
)

# Right: Total volume
fig_distribution.add_trace(
    go.Bar(
        x=bucket_order,
        y=bucket_volumes,
        marker=dict(color=COLOR_POSITIVE),
        hovertemplate='%{x}<br>Volume: $%{y:,.0f}<extra></extra>',
        showlegend=False
    ),
    row=1, col=2
)

fig_distribution.update_xaxes(title_text="Value Range", row=1, col=1, gridcolor=GRID_COLOR, color=TEXT_COLOR, tickangle=-45)
fig_distribution.update_xaxes(title_text="Value Range", row=1, col=2, gridcolor=GRID_COLOR, color=TEXT_COLOR, tickangle=-45)
fig_distribution.update_yaxes(title_text="Transfers", row=1, col=1, gridcolor=GRID_COLOR, color=TEXT_COLOR)
fig_distribution.update_yaxes(title_text="USD Volume", row=1, col=2, gridcolor=GRID_COLOR, color=TEXT_COLOR)

# Calculate statistics for annotation
median_val = df_with_usd_values['srcValueUsd'].median()
mean_val = df_with_usd_values['srcValueUsd'].mean()
p95_val = df_with_usd_values['srcValueUsd'].quantile(0.95)
total_volume = df_with_usd_values['srcValueUsd'].sum()

fig_distribution.update_layout(
    font=dict(size=FONT_SIZE_BASE, color=TEXT_COLOR),
    height=350,
    margin=dict(l=50, r=20, t=80, b=100),
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    title=dict(
        text=f'TRANSFER VALUE DISTRIBUTION<br><sub>Median: ${median_val:,.0f} | Mean: ${mean_val:,.0f} | Total Volume: ${total_volume/1e6:.1f}M</sub>',
        font=dict(size=FONT_SIZE_TITLE, color=TEXT_COLOR),
        x=0.5,
        xanchor='center'
    )
)

print(f"  Distribution: {len(df_with_usd_values)} transfers, median ${median_val:,.0f}, total ${total_volume/1e6:.1f}M")

# ========== STATS CARDS ==========
route_stats_summary = df.dropna(subset=['srcChain', 'dstChain']).groupby(['srcChain', 'dstChain']).size().sort_values(ascending=False)
top_route = route_stats_summary.index[0]
top_route_count = route_stats_summary.values[0]

avg_duration = df['duration'].mean()
fastest_plugin = plugin_stats.nsmallest(1, 'avg_duration_sec').iloc[0]

# Calculate additional stats
df_with_values = df.dropna(subset=['srcValueUsd'])
total_volume_usd = df_with_values['srcValueUsd'].sum()
median_transfer = df_with_values['srcValueUsd'].median()
if len(token_stats) > 0:
    top_token = token_stats.nlargest(1, 'transfer_count').iloc[0]
else:
    top_token = {'symbol': 'N/A', 'transfer_count': 0}

# Net flow winner
net_flow_winner = max(net_flows_sorted.items(), key=lambda x: x[1])
net_flow_loser = min(net_flows_sorted.items(), key=lambda x: x[1])

# ========== CREATE HTML ==========
print("Assembling HTML...")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interop Stats</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Courier New', monospace;
            background: {DARK_BG};
            color: {TEXT_COLOR};
            padding: 20px;
            font-size: 13px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid {GRID_COLOR};
        }}
        .header h1 {{
            font-size: 18px;
            font-weight: normal;
            color: {ACCENT_COLOR};
            letter-spacing: 2px;
        }}
        .header .timestamp {{
            font-size: 11px;
            color: {MUTED_TEXT};
        }}
        .stats-bar {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 10px;
            margin-bottom: 20px;
        }}
        .stat {{
            background: {CARD_BG};
            padding: 12px 15px;
            border-left: 2px solid {ACCENT_COLOR};
        }}
        .stat-label {{
            font-size: 10px;
            color: {MUTED_TEXT};
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .stat-value {{
            font-size: 20px;
            color: {ACCENT_COLOR};
            font-weight: bold;
            margin-top: 4px;
        }}
        .stat-sub {{
            font-size: 10px;
            color: {MUTED_TEXT};
            margin-top: 2px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 15px;
        }}
        .full-width {{
            grid-column: 1 / -1;
        }}
        .chart-container {{
            background: {CARD_BG};
            padding: 15px;
            border: 1px solid {GRID_COLOR};
            overflow: visible;
        }}
        .chart-container.sankey {{
            min-height: 600px;
            padding-bottom: 40px;
        }}
        .plugin-nav {{
            overflow-x: auto;
        }}

        /* Responsive design for tablets and smaller */
        @media (max-width: 1200px) {{
            .grid {{
                grid-template-columns: 1fr;
            }}
            .chart-container {{
                min-height: 400px;
            }}
            .header {{
                flex-direction: column;
                align-items: flex-start;
                gap: 10px;
            }}
        }}

        /* Responsive design for mobile */
        @media (max-width: 768px) {{
            body {{
                padding: 10px;
                font-size: 12px;
            }}
            .stats-bar {{
                grid-template-columns: repeat(2, 1fr);
                gap: 8px;
            }}
            .stat {{
                padding: 8px 10px;
            }}
            .stat-value {{
                font-size: 16px;
            }}
            .stat-label {{
                font-size: 9px;
            }}
            .stat-sub {{
                font-size: 9px;
            }}
            .header h1 {{
                font-size: 15px;
                letter-spacing: 1px;
            }}
            .header .timestamp {{
                font-size: 9px;
            }}
            .chart-container {{
                padding: 10px;
                margin-bottom: 10px;
            }}
            .chart-container.sankey {{
                min-height: 400px;
            }}
            .grid {{
                gap: 10px;
                margin-bottom: 10px;
            }}
            .plugin-nav {{
                padding: 15px;
                margin-top: 20px;
            }}
            .plugin-nav h2 {{
                font-size: 12px;
                margin-bottom: 10px;
            }}
        }}

        /* Responsive design for very small mobile */
        @media (max-width: 480px) {{
            body {{
                padding: 8px;
                font-size: 11px;
            }}
            .stats-bar {{
                grid-template-columns: 1fr;
                gap: 6px;
            }}
            .stat {{
                padding: 8px;
            }}
            .stat-value {{
                font-size: 14px;
            }}
            .stat-label {{
                font-size: 8px;
            }}
            .header h1 {{
                font-size: 14px;
            }}
            .chart-container {{
                padding: 8px;
            }}
            .plugin-nav {{
                padding: 10px;
                margin-top: 15px;
            }}
            .plugin-nav h2 {{
                font-size: 11px;
                margin-bottom: 8px;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⬡ INTEROP STATS</h1>
        <div class="timestamp">
            {date_range} | Last updated: {datetime.now().strftime('%H:%M:%S')}
        </div>
    </div>

    <div class="stats-bar">
        <div class="stat">
            <div class="stat-label">Total Volume</div>
            <div class="stat-value">${total_volume_usd/1e6:.1f}M</div>
            <div class="stat-sub">{total_transfers:,} transfers</div>
        </div>
        <div class="stat">
            <div class="stat-label">Median Transfer</div>
            <div class="stat-value">${median_transfer:.0f}</div>
            <div class="stat-sub">retail-dominated</div>
        </div>
        <div class="stat">
            <div class="stat-label">Top Token</div>
            <div class="stat-value">{top_token['symbol']}</div>
            <div class="stat-sub">{top_token['transfer_count']:,} transfers</div>
        </div>
        <div class="stat">
            <div class="stat-label">Top Route</div>
            <div class="stat-value">{top_route_count:,}</div>
            <div class="stat-sub">{top_route[0]} → {top_route[1]}</div>
        </div>
        <div class="stat">
            <div class="stat-label">Net Flow Winner</div>
            <div class="stat-value">{net_flow_winner[0]}</div>
            <div class="stat-sub">+${net_flow_winner[1]/1e3:.0f}K</div>
        </div>
        <div class="stat">
            <div class="stat-label">Fastest Protocol</div>
            <div class="stat-value">{fastest_plugin['avg_duration_sec']:.1f}s</div>
            <div class="stat-sub">{fastest_plugin['plugin']}</div>
        </div>
    </div>

    <div class="grid">
        <div class="chart-container sankey">
            <div id="chart-sankey-count"></div>
        </div>

        <div class="chart-container sankey">
            <div id="chart-sankey-usd"></div>
        </div>

        <div class="chart-container">
            <div id="chart-performance"></div>
        </div>

        <div class="chart-container">
            <div id="chart-net-flows"></div>
        </div>

        <div class="chart-container">
            <div id="chart-timeline-count"></div>
        </div>

        <div class="chart-container">
            <div id="chart-timeline-usd"></div>
        </div>

        <div class="chart-container full-width">
            <div id="chart-route-comparison"></div>
        </div>

        <div class="chart-container">
            <div id="chart-heatmap"></div>
        </div>

        <div class="chart-container">
            <div id="chart-protocol-size"></div>
        </div>

        <div class="chart-container">
            <div id="chart-tokens"></div>
        </div>

        <div class="chart-container">
            <div id="chart-chain-tokens"></div>
        </div>

        <div class="chart-container full-width">
            <div id="chart-distribution"></div>
        </div>

        <div class="chart-container full-width">
            <div id="chart-missing"></div>
        </div>
    </div>

    <div style="margin-top: 30px; padding: 20px; background: {CARD_BG}; border: 1px solid {GRID_COLOR};" class="plugin-nav">
        <h2 style="color: {ACCENT_COLOR}; font-size: 14px; margin-bottom: 15px; letter-spacing: 2px;">PLUGIN DETAIL PAGES</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px;">
            __PLUGIN_LINKS__
        </div>
    </div>

    <script>
        const config = {{
            displayModeBar: false,
            responsive: true
        }};

        Plotly.newPlot('chart-sankey-count', __SANKEY_COUNT_DATA__, __SANKEY_COUNT_LAYOUT__, config);
        Plotly.newPlot('chart-sankey-usd', __SANKEY_USD_DATA__, __SANKEY_USD_LAYOUT__, config);
        Plotly.newPlot('chart-performance', __PERFORMANCE_DATA__, __PERFORMANCE_LAYOUT__, config);
        Plotly.newPlot('chart-heatmap', __HEATMAP_DATA__, __HEATMAP_LAYOUT__, config);
        Plotly.newPlot('chart-timeline-count', __TIMELINE_COUNT_DATA__, __TIMELINE_COUNT_LAYOUT__, config);
        Plotly.newPlot('chart-timeline-usd', __TIMELINE_USD_DATA__, __TIMELINE_USD_LAYOUT__, config);
        Plotly.newPlot('chart-route-comparison', __ROUTE_COMPARISON_DATA__, __ROUTE_COMPARISON_LAYOUT__, config);
        Plotly.newPlot('chart-net-flows', __NET_FLOWS_DATA__, __NET_FLOWS_LAYOUT__, config);
        Plotly.newPlot('chart-protocol-size', __PROTOCOL_SIZE_DATA__, __PROTOCOL_SIZE_LAYOUT__, config);
        Plotly.newPlot('chart-tokens', __TOKENS_DATA__, __TOKENS_LAYOUT__, config);
        Plotly.newPlot('chart-chain-tokens', __CHAIN_TOKENS_DATA__, __CHAIN_TOKENS_LAYOUT__, config);
        Plotly.newPlot('chart-distribution', __DISTRIBUTION_DATA__, __DISTRIBUTION_LAYOUT__, config);
        Plotly.newPlot('chart-missing', __MISSING_DATA__, __MISSING_LAYOUT__, config);
    </script>
</body>
</html>
"""

# Generate plugin links
plugin_counts = df.groupby('plugin').size().sort_values(ascending=False)
plugin_links_html = ""
for plugin_name, count in plugin_counts.items():
    plugin_links_html += f'''
        <a href="plugins/{plugin_name}.html" style="display: block; padding: 12px 15px; background: {DARK_BG}; border: 1px solid {GRID_COLOR}; color: {TEXT_COLOR}; text-decoration: none; border-left: 2px solid {ACCENT_COLOR}; transition: all 0.2s;">
            <div style="font-size: 12px; font-weight: bold; color: {ACCENT_COLOR};">{plugin_name}</div>
            <div style="font-size: 10px; color: {MUTED_TEXT}; margin-top: 4px;">{count:,} transfers →</div>
        </a>
    '''

html_content = html_content.replace('__PLUGIN_LINKS__', plugin_links_html)

# Replace plotly JSON with proper data and layout
sankey_count_json = json.loads(pio.to_json(fig_sankey_count))
html_content = html_content.replace('__SANKEY_COUNT_DATA__', json.dumps(sankey_count_json['data']))
html_content = html_content.replace('__SANKEY_COUNT_LAYOUT__', json.dumps(sankey_count_json['layout']))

sankey_usd_json = json.loads(pio.to_json(fig_sankey_usd))
html_content = html_content.replace('__SANKEY_USD_DATA__', json.dumps(sankey_usd_json['data']))
html_content = html_content.replace('__SANKEY_USD_LAYOUT__', json.dumps(sankey_usd_json['layout']))

performance_json = json.loads(pio.to_json(fig_performance))
html_content = html_content.replace('__PERFORMANCE_DATA__', json.dumps(performance_json['data']))
html_content = html_content.replace('__PERFORMANCE_LAYOUT__', json.dumps(performance_json['layout']))

heatmap_json = json.loads(pio.to_json(fig_heatmap))
html_content = html_content.replace('__HEATMAP_DATA__', json.dumps(heatmap_json['data']))
html_content = html_content.replace('__HEATMAP_LAYOUT__', json.dumps(heatmap_json['layout']))

route_comparison_json = json.loads(pio.to_json(fig_route_comparison))
html_content = html_content.replace('__ROUTE_COMPARISON_DATA__', json.dumps(route_comparison_json['data']))
html_content = html_content.replace('__ROUTE_COMPARISON_LAYOUT__', json.dumps(route_comparison_json['layout']))

timeline_count_json = json.loads(pio.to_json(fig_timeline_count))
html_content = html_content.replace('__TIMELINE_COUNT_DATA__', json.dumps(timeline_count_json['data']))
html_content = html_content.replace('__TIMELINE_COUNT_LAYOUT__', json.dumps(timeline_count_json['layout']))

timeline_usd_json = json.loads(pio.to_json(fig_timeline_usd))
html_content = html_content.replace('__TIMELINE_USD_DATA__', json.dumps(timeline_usd_json['data']))
html_content = html_content.replace('__TIMELINE_USD_LAYOUT__', json.dumps(timeline_usd_json['layout']))

net_flows_json = json.loads(pio.to_json(fig_net_flows))
html_content = html_content.replace('__NET_FLOWS_DATA__', json.dumps(net_flows_json['data']))
html_content = html_content.replace('__NET_FLOWS_LAYOUT__', json.dumps(net_flows_json['layout']))

protocol_size_json = json.loads(pio.to_json(fig_protocol_size))
html_content = html_content.replace('__PROTOCOL_SIZE_DATA__', json.dumps(protocol_size_json['data']))
html_content = html_content.replace('__PROTOCOL_SIZE_LAYOUT__', json.dumps(protocol_size_json['layout']))

tokens_json = json.loads(pio.to_json(fig_tokens))
html_content = html_content.replace('__TOKENS_DATA__', json.dumps(tokens_json['data']))
html_content = html_content.replace('__TOKENS_LAYOUT__', json.dumps(tokens_json['layout']))

chain_tokens_json = json.loads(pio.to_json(fig_chain_tokens))
html_content = html_content.replace('__CHAIN_TOKENS_DATA__', json.dumps(chain_tokens_json['data']))
html_content = html_content.replace('__CHAIN_TOKENS_LAYOUT__', json.dumps(chain_tokens_json['layout']))

distribution_json = json.loads(pio.to_json(fig_distribution))
html_content = html_content.replace('__DISTRIBUTION_DATA__', json.dumps(distribution_json['data']))
html_content = html_content.replace('__DISTRIBUTION_LAYOUT__', json.dumps(distribution_json['layout']))

missing_json = json.loads(pio.to_json(fig_missing))
html_content = html_content.replace('__MISSING_DATA__', json.dumps(missing_json['data']))
html_content = html_content.replace('__MISSING_LAYOUT__', json.dumps(missing_json['layout']))

with open('index.html', 'w') as f:
    f.write(html_content)

print("✓ Dashboard created")
print(f"  {total_transfers:,} transfers | {total_plugins} protocols | {total_chains} chains")
