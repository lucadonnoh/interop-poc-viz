import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import json
import plotly.io as pio
import math
import numpy as np
import os

# Read the data
df = pd.read_csv('interop_fresh.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Import styling constants from main dashboard
DARK_BG = '#0a0e27'
CARD_BG = '#111836'
TEXT_COLOR = '#ffffff'
ACCENT_COLOR = '#00d9ff'
GRID_COLOR = '#1a2332'
MUTED_TEXT = '#a0aec0'

FONT_SIZE_BASE = 12
FONT_SIZE_AXIS = 10
FONT_SIZE_LEGEND = 10
FONT_SIZE_ANNOTATION = 9
FONT_SIZE_TITLE = 14

COLOR_INFO = 'rgba(99, 110, 250, 0.8)'
COLOR_POSITIVE = 'rgba(0, 204, 150, 0.8)'

def create_plugin_page(plugin_name, df, all_plugins):
    """Generate a detailed analysis page for a specific plugin"""

    print(f"  Generating page for {plugin_name}...")

    # Filter data for this plugin
    plugin_df = df[df['plugin'] == plugin_name].copy()
    total_transfers = len(plugin_df)

    if total_transfers == 0:
        print(f"    Skipping {plugin_name} - no transfers")
        return

    # ========== STATS ==========
    plugin_with_usd = plugin_df.dropna(subset=['srcValueUsd'])
    total_volume_usd = plugin_with_usd['srcValueUsd'].sum() if len(plugin_with_usd) > 0 else 0
    avg_transfer_usd = plugin_with_usd['srcValueUsd'].mean() if len(plugin_with_usd) > 0 else 0
    median_transfer_usd = plugin_with_usd['srcValueUsd'].median() if len(plugin_with_usd) > 0 else 0

    plugin_with_duration = plugin_df.dropna(subset=['duration'])
    avg_duration = plugin_with_duration['duration'].mean() if len(plugin_with_duration) > 0 else 0
    median_duration = plugin_with_duration['duration'].median() if len(plugin_with_duration) > 0 else 0
    p95_duration = plugin_with_duration['duration'].quantile(0.95) if len(plugin_with_duration) > 0 else 0

    # Most popular route
    route_counts = plugin_df.dropna(subset=['srcChain', 'dstChain']).groupby(['srcChain', 'dstChain']).size()
    top_route = route_counts.idxmax() if len(route_counts) > 0 else ('N/A', 'N/A')
    top_route_count = route_counts.max() if len(route_counts) > 0 else 0

    # Most popular token
    token_counts = plugin_df.dropna(subset=['srcAbstractTokenId']).groupby('srcAbstractTokenId').size()
    if len(token_counts) > 0:
        top_token_id = token_counts.idxmax()
        top_token_symbol = top_token_id.split(':')[-1] if ':' in top_token_id else top_token_id
        top_token_count = token_counts.max()
    else:
        top_token_symbol = 'N/A'
        top_token_count = 0

    # ========== SANKEY DIAGRAMS ==========
    # Sankey by transfer count
    df_flow = plugin_df.dropna(subset=['srcChain', 'dstChain'])
    all_chains = sorted(list(set(df_flow['srcChain'].unique()) | set(df_flow['dstChain'].unique())))
    chain_to_idx = {chain: idx for idx, chain in enumerate(all_chains)}
    num_chains = len(all_chains)

    flow_data_count = df_flow.groupby(['srcChain', 'dstChain']).size().reset_index(name='count')
    sources_count, targets_count, values_count = [], [], []
    for _, row in flow_data_count.iterrows():
        sources_count.append(chain_to_idx[row['srcChain']])  # Left side
        targets_count.append(chain_to_idx[row['dstChain']] + num_chains)  # Right side
        values_count.append(row['count'])

    # Calculate node statistics for transfer count
    # Left side nodes (sources)
    node_stats_count = []
    for chain in all_chains:
        outgoing = df_flow[df_flow['srcChain'] == chain]
        out_count = len(outgoing)
        node_stats_count.append(
            f"<b>{chain} (Source)</b><br>Outgoing: {out_count:,} transfers"
        )

    # Right side nodes (destinations)
    for chain in all_chains:
        incoming = df_flow[df_flow['dstChain'] == chain]
        in_count = len(incoming)
        node_stats_count.append(
            f"<b>{chain} (Destination)</b><br>Incoming: {in_count:,} transfers"
        )

    fig_sankey_count = go.Figure(data=[go.Sankey(
        node=dict(
            pad=30,
            thickness=20,
            line=dict(color=DARK_BG, width=1),
            label=all_chains + all_chains,
            color=[ACCENT_COLOR] * (num_chains * 2),
            customdata=node_stats_count,
            hovertemplate='%{customdata}<extra></extra>',
            x=[0.01] * num_chains + [0.99] * num_chains
        ),
        link=dict(
            source=sources_count,
            target=targets_count,
            value=values_count,
            color=['rgba(0, 217, 255, 0.3)'] * len(values_count),
            hovertemplate='%{source.label} → %{target.label}<br>Transfers: %{value:,}<extra></extra>'
        )
    )])

    fig_sankey_count.update_layout(
        font=dict(size=FONT_SIZE_BASE, color=TEXT_COLOR),
        height=400,
        margin=dict(l=80, r=80, t=80, b=20),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        title=dict(text='NETWORK FLOW (Transfer Count)', font=dict(size=FONT_SIZE_TITLE, color=TEXT_COLOR), x=0.5, xanchor='center'),
        annotations=[
            dict(x=0.01, y=1.05, xref='paper', yref='paper', text='<b>Source</b>', showarrow=False, font=dict(size=FONT_SIZE_AXIS, color=TEXT_COLOR), xanchor='left', align='left'),
            dict(x=0.99, y=1.05, xref='paper', yref='paper', text='<b>Destination</b>', showarrow=False, font=dict(size=FONT_SIZE_AXIS, color=TEXT_COLOR), xanchor='right', align='right')
        ]
    )

    # Sankey by USD volume (broken down by token)
    df_flow_usd = plugin_df.dropna(subset=['srcChain', 'dstChain', 'srcValueUsd', 'srcAbstractTokenId']).copy()
    df_flow_usd['token_symbol'] = df_flow_usd['srcAbstractTokenId'].str.split(':').str[-1]

    flow_data_usd = df_flow_usd.groupby(['srcChain', 'dstChain', 'token_symbol'])['srcValueUsd'].sum().reset_index(name='volume')

    # Create temporary global token order for this plugin (before the main one is created)
    temp_token_stats = plugin_df.dropna(subset=['srcAbstractTokenId']).groupby('srcAbstractTokenId').size().reset_index(name='count')
    temp_token_stats['symbol'] = temp_token_stats['srcAbstractTokenId'].str.split(':').str[-1]
    temp_token_stats = temp_token_stats.sort_values('count', ascending=False)
    temp_token_order = temp_token_stats['symbol'].tolist()

    # Define brand-accurate colors for tokens (same as scatter plot/breakdown)
    TOKEN_BRAND_COLORS_SANKEY = {
        'USDC': 'rgba(60, 145, 230, 0.5)',      # USDC Blue (brighter)
        'ETH': 'rgba(150, 100, 255, 0.5)',      # Ethereum Purple (more purple, less blue)
        'USDT': 'rgba(50, 175, 135, 0.5)',      # Tether Green (brighter)
        'USDT0': 'rgba(70, 200, 160, 0.5)',     # Tether Green lighter (brighter)
        'WBTC': 'rgba(247, 147, 26, 0.5)',      # Bitcoin Orange
        'DAI': 'rgba(255, 184, 77, 0.5)',       # DAI Gold/Yellow
        'ZRO': 'rgba(170, 140, 255, 0.5)',      # LayerZero Purple (brighter)
        'SNX': 'rgba(0, 209, 255, 0.5)',        # Synthetix Cyan
        'POOL': 'rgba(100, 70, 160, 0.5)',      # PoolTogether Purple (brighter)
        'ACX': 'rgba(255, 88, 88, 0.5)',        # Across Red
        'WLD': 'rgba(80, 80, 80, 0.5)',         # Worldcoin Gray (brighter)
        'VLR': 'rgba(120, 220, 180, 0.5)',      # Generic teal
    }
    sankey_fallback_palette = px.colors.qualitative.Set3

    def get_sankey_token_color(token_symbol, index_fallback):
        """Get color for a token in sankey - use brand color with lower opacity"""
        if token_symbol in TOKEN_BRAND_COLORS_SANKEY:
            return TOKEN_BRAND_COLORS_SANKEY[token_symbol]
        else:
            # Convert Set3 color to rgba with lower opacity
            base_color = sankey_fallback_palette[index_fallback % len(sankey_fallback_palette)]
            if base_color.startswith('rgb('):
                return base_color.replace('rgb(', 'rgba(').replace(')', ', 0.5)')
            return base_color

    # Get top tokens and assign colors
    token_colors = {}
    top_tokens = df_flow_usd.groupby('token_symbol')['srcValueUsd'].sum().nlargest(10)
    for token in top_tokens.index:
        if token in temp_token_order:
            token_colors[token] = get_sankey_token_color(token, temp_token_order.index(token))
        else:
            token_colors[token] = 'rgba(128, 128, 128, 0.3)'
    default_color = 'rgba(128, 128, 128, 0.3)'

    sources_usd, targets_usd, values_usd, colors_usd, labels_usd = [], [], [], [], []
    for _, row in flow_data_usd.iterrows():
        sources_usd.append(chain_to_idx[row['srcChain']])  # Left side
        targets_usd.append(chain_to_idx[row['dstChain']] + num_chains)  # Right side
        values_usd.append(row['volume'])
        colors_usd.append(token_colors.get(row['token_symbol'], default_color))
        labels_usd.append(row['token_symbol'])

    # Calculate node statistics for USD volume
    # Left side nodes (sources)
    node_stats_usd = []
    for chain in all_chains:
        outgoing = df_flow_usd[df_flow_usd['srcChain'] == chain]
        out_volume = outgoing['srcValueUsd'].sum()
        top_token_out = outgoing.groupby('token_symbol')['srcValueUsd'].sum().idxmax() if len(outgoing) > 0 else 'N/A'
        node_stats_usd.append(
            f"<b>{chain} (Source)</b><br>Outgoing: ${out_volume:,.0f}<br>Top token: {top_token_out}"
        )

    # Right side nodes (destinations)
    for chain in all_chains:
        incoming = df_flow_usd[df_flow_usd['dstChain'] == chain]
        in_volume = incoming['srcValueUsd'].sum()
        top_token_in = incoming.groupby('token_symbol')['srcValueUsd'].sum().idxmax() if len(incoming) > 0 else 'N/A'
        node_stats_usd.append(
            f"<b>{chain} (Destination)</b><br>Incoming: ${in_volume:,.0f}<br>Top token: {top_token_in}"
        )

    fig_sankey_usd = go.Figure(data=[go.Sankey(
        domain=dict(x=[0, 0.82], y=[0, 1]),  # Constrain Sankey to left 82% of figure
        node=dict(
            pad=30,
            thickness=20,
            line=dict(color=DARK_BG, width=1),
            label=all_chains + all_chains,
            color=[ACCENT_COLOR] * (num_chains * 2),
            customdata=node_stats_usd,
            hovertemplate='%{customdata}<extra></extra>',
            x=[0.01] * num_chains + [0.99] * num_chains
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

    # Build legend annotations for tokens
    legend_annotations = [
        dict(x=0.01, y=1.05, xref='paper', yref='paper', text='<b>Source</b>', showarrow=False, font=dict(size=FONT_SIZE_AXIS, color=TEXT_COLOR), xanchor='left', align='left'),
        dict(x=0.99, y=1.05, xref='paper', yref='paper', text='<b>Destination</b>', showarrow=False, font=dict(size=FONT_SIZE_AXIS, color=TEXT_COLOR), xanchor='right', align='right'),
        dict(x=0.98, y=0.98, xref='paper', yref='paper', text='<b>Tokens</b>', showarrow=False, font=dict(size=FONT_SIZE_AXIS, color=TEXT_COLOR), xanchor='right', align='right')
    ]

    # Add token color legend items
    for i, (token, color) in enumerate(token_colors.items()):
        legend_annotations.append(
            dict(
                x=0.98, y=0.93 - i*0.055, xref='paper', yref='paper',
                text=f'<span style="color:{color}">■</span> {token}',
                showarrow=False,
                font=dict(size=FONT_SIZE_ANNOTATION, color=TEXT_COLOR),
                xanchor='right', align='right'
            )
        )

    fig_sankey_usd.update_layout(
        font=dict(size=FONT_SIZE_BASE, color=TEXT_COLOR),
        height=400,
        margin=dict(l=80, r=20, t=80, b=20),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        title=dict(text='NETWORK FLOW (USD Volume)', font=dict(size=FONT_SIZE_TITLE, color=TEXT_COLOR), x=0.5, xanchor='center'),
        annotations=legend_annotations
    )

    # ========== Create Global Token Order (for consistent colors across charts) ==========
    # This will be used by both scatter plot and token breakdown to ensure consistent colors
    global_token_stats = plugin_df.dropna(subset=['srcAbstractTokenId']).groupby('srcAbstractTokenId').size().reset_index(name='count')
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
    token_color_palette_fallback = px.colors.qualitative.Set3

    def get_token_color(token_symbol, index_fallback):
        """Get color for a token - use brand color if available, otherwise use palette"""
        if token_symbol in TOKEN_BRAND_COLORS:
            return TOKEN_BRAND_COLORS[token_symbol]
        else:
            return token_color_palette_fallback[index_fallback % len(token_color_palette_fallback)]

    # ========== CHART 1: Duration vs Size Scatter ==========
    scatter_df = plugin_df.dropna(subset=['srcValueUsd', 'duration']).copy()

    # Extract token symbol from srcAbstractTokenId
    scatter_df['token_symbol'] = scatter_df['srcAbstractTokenId'].apply(
        lambda x: x.split(':')[-1] if pd.notna(x) else 'Unknown'
    )

    # Prepare custom data for hover with dynamic decimal places
    def format_amount(x):
        if pd.isna(x):
            return 'N/A'
        if x >= 1:
            return f"{x:,.2f}"  # 2 decimals for amounts >= 1
        elif x >= 0.01:
            return f"{x:,.4f}"  # 4 decimals for amounts >= 0.01
        elif x > 0:
            return f"{x:,.8f}"  # 8 decimals for very small amounts
        else:
            return '0'

    def format_duration(seconds):
        if pd.isna(seconds):
            return 'N/A'
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            # Show minutes and seconds
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        elif seconds < 86400:
            # Show hours and minutes
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"
        else:
            # Show days and hours
            days = int(seconds // 86400)
            hours = int((seconds % 86400) // 3600)
            return f"{days}d {hours}h"

    scatter_df['srcAmount_display'] = scatter_df['srcAmount'].apply(format_amount)
    scatter_df['duration_display'] = scatter_df['duration'].apply(format_duration)

    fig_duration_vs_size = go.Figure()

    if len(scatter_df) > 0:
        # Get unique tokens in scatter data, but order them according to global token order
        scatter_tokens_set = set(scatter_df['token_symbol'].unique())
        unique_tokens = [token for token in global_token_order if token in scatter_tokens_set]

        # Assign colors using brand colors or fallback palette
        token_colors = {token: get_token_color(token, global_token_order.index(token))
                       for token in unique_tokens}

        # Create one trace per token
        for token in unique_tokens:
            token_data = scatter_df[scatter_df['token_symbol'] == token]

            fig_duration_vs_size.add_trace(go.Scatter(
                x=token_data['srcValueUsd'].tolist(),
                y=token_data['duration'].tolist(),
                mode='markers',
                name=token,
                marker=dict(
                    size=5,
                    color=token_colors[token],
                    opacity=0.6,
                    line=dict(width=0)
                ),
                customdata=list(zip(
                    token_data['srcAmount_display'].tolist(),
                    token_data['token_symbol'].tolist(),
                    token_data['duration_display'].tolist()
                )),
                hovertemplate='<b>Transfer</b><br>Value: $%{x:,.0f}<br>Duration: %{customdata[2]}<br>Amount: %{customdata[0]} %{customdata[1]}<extra></extra>',
                showlegend=True
            ))

        # Add trend line (in log-log space)
        if len(scatter_df) > 10:
            try:
                # Fit line in log-log space
                log_x = np.log10(scatter_df['srcValueUsd'])
                log_y = np.log10(scatter_df['duration'])

                # Filter out any inf or NaN values
                valid_mask = np.isfinite(log_x) & np.isfinite(log_y)
                log_x_valid = log_x[valid_mask]
                log_y_valid = log_y[valid_mask]

                # Only fit if we have enough valid points and some variation
                if len(log_x_valid) > 10 and log_x_valid.std() > 0 and log_y_valid.std() > 0:
                    z = np.polyfit(log_x_valid, log_y_valid, 1)
                    p = np.poly1d(z)

                    # Generate trend line points using valid data range
                    x_min = 10 ** log_x_valid.min()
                    x_max = 10 ** log_x_valid.max()
                    x_trend = np.logspace(np.log10(x_min), np.log10(x_max), 50)
                    y_trend = 10 ** p(np.log10(x_trend))

                    fig_duration_vs_size.add_trace(go.Scatter(
                        x=x_trend.tolist(),
                        y=y_trend.tolist(),
                        mode='lines',
                        line=dict(color='rgba(255, 107, 107, 0.8)', width=2, dash='dash'),
                        name='Trend',
                        hoverinfo='skip',
                        showlegend=False
                    ))
            except (np.linalg.LinAlgError, ValueError):
                # Skip trend line if fitting fails
                pass

    fig_duration_vs_size.update_layout(
        font=dict(size=FONT_SIZE_BASE, color=TEXT_COLOR),
        xaxis=dict(
            title='Transfer Value (USD)',
            type='log',
            gridcolor=GRID_COLOR,
            color=TEXT_COLOR
        ),
        yaxis=dict(
            title='Duration (seconds)',
            type='log',
            gridcolor=GRID_COLOR,
            color=TEXT_COLOR
        ),
        height=400,
        margin=dict(l=60, r=120, t=80, b=60),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            font=dict(size=FONT_SIZE_LEGEND),
            orientation='v',
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top'
        ),
        title=dict(
            text=f'TRANSFER DURATION VS SIZE<br><sub>Does transfer size affect speed?</sub>',
            font=dict(size=FONT_SIZE_TITLE, color=TEXT_COLOR),
            x=0.5,
            xanchor='center'
        )
    )

    # ========== CHART 2: Transfer Size Distribution ==========
    size_df = plugin_df.dropna(subset=['srcValueUsd'])

    # Define value buckets using same function as main dashboard
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

    size_df = size_df.copy()
    size_df['bucket'] = size_df['srcValueUsd'].apply(categorize_value)

    # Define bucket order
    bucket_order = ['<$10', '$10-$50', '$50-$100', '$100-$500', '$500-$1K',
                    '$1K-$5K', '$5K-$10K', '$10K-$50K', '$50K-$100K', '>$100K']

    # Count transfers and sum volume per bucket
    bucket_stats = size_df.groupby('bucket')['srcValueUsd'].agg(['count', 'sum'])
    bucket_counts = [bucket_stats.loc[bucket, 'count'] if bucket in bucket_stats.index else 0 for bucket in bucket_order]
    bucket_volumes = [bucket_stats.loc[bucket, 'sum'] if bucket in bucket_stats.index else 0 for bucket in bucket_order]

    # Create side-by-side charts
    fig_size_dist = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.12
    )

    # Left: Transfer count
    fig_size_dist.add_trace(
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
    fig_size_dist.add_trace(
        go.Bar(
            x=bucket_order,
            y=bucket_volumes,
            marker=dict(color=COLOR_POSITIVE),
            hovertemplate='%{x}<br>Volume: $%{y:,.0f}<extra></extra>',
            showlegend=False
        ),
        row=1, col=2
    )

    fig_size_dist.update_xaxes(title_text="Value Range", row=1, col=1, gridcolor=GRID_COLOR, color=TEXT_COLOR, tickangle=-45)
    fig_size_dist.update_xaxes(title_text="Value Range", row=1, col=2, gridcolor=GRID_COLOR, color=TEXT_COLOR, tickangle=-45)
    fig_size_dist.update_yaxes(title_text="Transfers", row=1, col=1, gridcolor=GRID_COLOR, color=TEXT_COLOR)
    fig_size_dist.update_yaxes(title_text="USD Volume", row=1, col=2, gridcolor=GRID_COLOR, color=TEXT_COLOR)

    total_volume = size_df['srcValueUsd'].sum()

    fig_size_dist.update_layout(
        font=dict(size=FONT_SIZE_BASE, color=TEXT_COLOR),
        height=350,
        margin=dict(l=50, r=20, t=80, b=100),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        title=dict(
            text=f'TRANSFER VALUE DISTRIBUTION<br><sub>Median: ${median_transfer_usd:,.0f} | Mean: ${avg_transfer_usd:,.0f} | Total Volume: ${total_volume/1e6:.1f}M</sub>',
            font=dict(size=FONT_SIZE_TITLE, color=TEXT_COLOR),
            x=0.5,
            xanchor='center'
        )
    )

    # ========== CHART 3: Route Performance Heatmap ==========
    route_df = plugin_df.dropna(subset=['srcChain', 'dstChain', 'duration'])
    route_stats = route_df.groupby(['srcChain', 'dstChain']).agg({
        'duration': 'mean',
        'transferId': 'count'
    }).reset_index()
    route_stats.columns = ['srcChain', 'dstChain', 'avg_duration', 'transfer_count']

    fig_route_heatmap = go.Figure()

    if len(route_stats) > 0:
        all_chains = sorted(list(set(route_stats['srcChain'].unique()) | set(route_stats['dstChain'].unique())))
        chain_to_idx = {chain: idx for idx, chain in enumerate(all_chains)}

        duration_matrix = np.zeros((len(all_chains), len(all_chains)))
        count_matrix = np.zeros((len(all_chains), len(all_chains)))

        for _, row in route_stats.iterrows():
            duration_matrix[chain_to_idx[row['srcChain']]][chain_to_idx[row['dstChain']]] = row['avg_duration']
            count_matrix[chain_to_idx[row['srcChain']]][chain_to_idx[row['dstChain']]] = row['transfer_count']

        # Create hover text
        hover_text = []
        for i, src in enumerate(all_chains):
            row_text = []
            for j, dst in enumerate(all_chains):
                count = int(count_matrix[i][j])
                dur = duration_matrix[i][j]
                if count > 0:
                    row_text.append(f'<b>{src} → {dst}</b><br>Duration: {dur:.1f}s<br>Count: {count:,}')
                else:
                    row_text.append(f'{src} → {dst}<br>No transfers')
            hover_text.append(row_text)

        fig_route_heatmap.add_trace(go.Heatmap(
            z=duration_matrix.tolist(),
            x=all_chains,
            y=all_chains,
            colorscale='Viridis',
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_text,
            colorbar=dict(title=dict(text='sec', side='right', font=dict(size=FONT_SIZE_AXIS, color=TEXT_COLOR)))
        ))

    fig_route_heatmap.update_layout(
        font=dict(size=FONT_SIZE_BASE, color=TEXT_COLOR),
        xaxis=dict(title='Destination', color=TEXT_COLOR),
        yaxis=dict(title='Source', color=TEXT_COLOR),
        height=350,
        margin=dict(l=60, r=60, t=80, b=60),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        title=dict(
            text='ROUTE PERFORMANCE',
            font=dict(size=FONT_SIZE_TITLE, color=TEXT_COLOR),
            x=0.5,
            xanchor='center'
        )
    )

    # ========== CHART 4: Token Breakdown ==========
    token_df = plugin_df.dropna(subset=['srcAbstractTokenId', 'srcValueUsd'])

    # Aggregate by token for both count and volume
    token_stats = token_df.groupby('srcAbstractTokenId').agg({
        'transferId': 'count',
        'srcValueUsd': 'sum'
    }).reset_index()
    token_stats.columns = ['token', 'count', 'volume']
    token_stats['symbol'] = token_stats['token'].str.split(':').str[-1]

    # Calculate total for percentages
    total_count = token_stats['count'].sum()
    total_volume = token_stats['volume'].sum()

    # Calculate percentages
    token_stats['count_pct'] = (token_stats['count'] / total_count) * 100
    token_stats['volume_pct'] = (token_stats['volume'] / total_volume) * 100

    # Filter out tokens with less than 0.1% and aggregate into "Other"
    # For count
    tokens_count_main = token_stats[token_stats['count_pct'] >= 0.1].copy()
    tokens_count_other = token_stats[token_stats['count_pct'] < 0.1].copy()

    # For volume
    tokens_volume_main = token_stats[token_stats['volume_pct'] >= 0.1].copy()
    tokens_volume_other = token_stats[token_stats['volume_pct'] < 0.1].copy()

    # Create side-by-side donut charts
    fig_token_breakdown = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'domain'}, {'type': 'domain'}]],
        subplot_titles=['By Transfer Count (≥0.1%)', 'By USD Volume (≥0.1%)'],
        horizontal_spacing=0.15
    )

    if len(token_stats) > 0:
        # Left: Transfer count - create explicit color mapping
        count_labels = tokens_count_main['symbol'].tolist()
        count_values = tokens_count_main['count'].tolist()

        # Create explicit color list based on brand colors or global token order
        count_colors = [get_token_color(label, global_token_order.index(label))
                       for label in count_labels]

        if len(tokens_count_other) > 0:
            count_labels.append('Other')
            count_values.append(tokens_count_other['count'].sum())
            count_colors.append('rgba(200, 200, 200, 0.8)')  # Gray for "Other"

        fig_token_breakdown.add_trace(go.Pie(
            labels=count_labels,
            values=count_values,
            hole=0.4,
            marker=dict(colors=count_colors),
            textinfo='label+percent',
            textfont=dict(size=FONT_SIZE_AXIS, color='white'),
            hovertemplate='<b>%{label}</b><br>%{value:,} transfers<br>%{percent}<extra></extra>'
        ), row=1, col=1)

        # Right: USD volume - create explicit color mapping
        volume_labels = tokens_volume_main['symbol'].tolist()
        volume_values = tokens_volume_main['volume'].tolist()

        # Create explicit color list based on brand colors or global token order
        volume_colors = [get_token_color(label, global_token_order.index(label))
                        for label in volume_labels]

        if len(tokens_volume_other) > 0:
            volume_labels.append('Other')
            volume_values.append(tokens_volume_other['volume'].sum())
            volume_colors.append('rgba(200, 200, 200, 0.8)')  # Gray for "Other"

        fig_token_breakdown.add_trace(go.Pie(
            labels=volume_labels,
            values=volume_values,
            hole=0.4,
            marker=dict(colors=volume_colors),
            textinfo='label+percent',
            textfont=dict(size=FONT_SIZE_AXIS, color='white'),
            hovertemplate='<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>'
        ), row=1, col=2)

    # Update subplot title styling
    for annotation in fig_token_breakdown['layout']['annotations']:
        annotation['font'] = dict(size=FONT_SIZE_BASE, color=MUTED_TEXT)

    fig_token_breakdown.update_layout(
        font=dict(size=FONT_SIZE_BASE, color=TEXT_COLOR),
        height=350,
        margin=dict(l=20, r=20, t=100, b=20),
        paper_bgcolor=CARD_BG,
        showlegend=False,
        title=dict(
            text=f'TOKEN BREAKDOWN',
            font=dict(size=FONT_SIZE_TITLE, color=TEXT_COLOR),
            x=0.5,
            xanchor='center'
        )
    )

    # ========== CHART 5: Duration Distribution ==========
    duration_df = plugin_df.dropna(subset=['duration'])

    fig_duration_dist = go.Figure()

    if len(duration_df) > 0:
        bins = np.linspace(0, min(duration_df['duration'].quantile(0.95) * 1.5, duration_df['duration'].max()), 40)
        hist, bin_edges = np.histogram(duration_df['duration'], bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        fig_duration_dist.add_trace(go.Bar(
            x=bin_centers.tolist(),
            y=hist.tolist(),
            marker=dict(color='rgba(0, 204, 150, 0.7)'),
            hovertemplate='%{x:.1f}s<br>Transfers: %{y}<extra></extra>',
            showlegend=False
        ))

        # Add percentile lines
        p50 = duration_df['duration'].quantile(0.50)
        p95 = duration_df['duration'].quantile(0.95)

        fig_duration_dist.add_vline(x=p50, line_dash="dash", line_color="yellow", annotation_text="p50", annotation_position="top")
        fig_duration_dist.add_vline(x=p95, line_dash="dash", line_color="red", annotation_text="p95", annotation_position="top")

    fig_duration_dist.update_layout(
        font=dict(size=FONT_SIZE_BASE, color=TEXT_COLOR),
        xaxis=dict(
            title='Duration (seconds)',
            gridcolor=GRID_COLOR,
            color=TEXT_COLOR
        ),
        yaxis=dict(
            title='Number of Transfers',
            gridcolor=GRID_COLOR,
            color=TEXT_COLOR
        ),
        height=350,
        margin=dict(l=60, r=20, t=80, b=60),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        bargap=0.1,
        title=dict(
            text=f'DURATION DISTRIBUTION<br><sub>Median: {median_duration:.1f}s | p95: {p95_duration:.1f}s</sub>',
            font=dict(size=FONT_SIZE_TITLE, color=TEXT_COLOR),
            x=0.5,
            xanchor='center'
        )
    )

    # ========== CHART 6: Hourly Activity Pattern ==========
    activity_df = plugin_df.dropna(subset=['timestamp']).copy()

    fig_activity_pattern = go.Figure()

    if len(activity_df) > 0:
        # Group by hour with timestamp
        activity_df_time = activity_df.set_index('timestamp')
        hourly_counts = activity_df_time.resample('1h').size()

        # Filter out last incomplete hour
        if len(hourly_counts) > 0:
            last_hour = hourly_counts.index.max()
            hourly_counts = hourly_counts[hourly_counts.index < last_hour]

        fig_activity_pattern.add_trace(go.Bar(
            x=hourly_counts.index.strftime('%Y-%m-%d %H:%M').tolist(),
            y=hourly_counts.values.tolist(),
            marker=dict(color=ACCENT_COLOR),
            hovertemplate='<b>%{x}</b><br>Transfers: %{y:,}<extra></extra>',
            showlegend=False
        ))

    fig_activity_pattern.update_layout(
        font=dict(size=FONT_SIZE_BASE, color=TEXT_COLOR),
        xaxis=dict(
            title='Time (UTC)',
            color=TEXT_COLOR,
            tickangle=-45,
            gridcolor=GRID_COLOR
        ),
        yaxis=dict(
            title='Transfers per Hour',
            color=TEXT_COLOR,
            gridcolor=GRID_COLOR
        ),
        height=350,
        margin=dict(l=60, r=20, t=80, b=100),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        bargap=0.1,
        title=dict(
            text='HOURLY ACTIVITY PATTERN',
            font=dict(size=FONT_SIZE_TITLE, color=TEXT_COLOR),
            x=0.5,
            xanchor='center'
        )
    )

    # ========== CHART 7: Net USD Flows ==========
    net_flows_df = plugin_df.dropna(subset=['srcChain', 'dstChain', 'srcValueUsd'])

    # Calculate outgoing and incoming for each chain
    outgoing = net_flows_df.groupby('srcChain')['srcValueUsd'].sum()
    incoming = net_flows_df.groupby('dstChain')['srcValueUsd'].sum()

    # Calculate net flows
    net_flows = {}
    for chain in all_chains:
        out = outgoing.get(chain, 0)
        in_val = incoming.get(chain, 0)
        net_flows[chain] = in_val - out

    # Sort by net flow value
    net_flows_sorted = dict(sorted(net_flows.items(), key=lambda x: x[1]))

    # Prepare data for horizontal bar chart
    chains = list(net_flows_sorted.keys())
    net_values = list(net_flows_sorted.values())
    colors_net = [COLOR_POSITIVE if v > 0 else 'rgba(239, 85, 59, 0.8)' for v in net_values]

    # Text labels outside bars
    text_labels = [f"${abs(v)/1e3:.0f}K" if abs(v) >= 1000 else f"${abs(v):.0f}" for v in net_values]

    fig_net_flows = go.Figure(data=[go.Bar(
        y=chains,
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
            automargin=True
        ),
        yaxis=dict(
            title='',
            color=TEXT_COLOR,
            tickfont=dict(size=FONT_SIZE_AXIS),
            automargin=True,
            ticksuffix="        "
        ),
        height=300,
        margin=dict(l=100, r=120, t=60, b=50, pad=10, autoexpand=True),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        showlegend=False,
        title=dict(text='NET USD FLOWS', font=dict(size=FONT_SIZE_TITLE, color=TEXT_COLOR), x=0.5, xanchor='center'),
        bargap=0.3,
        uniformtext_minsize=FONT_SIZE_ANNOTATION,
        uniformtext_mode='show'
    )

    # Convert to JSON for embedding
    sankey_count_json = json.loads(pio.to_json(fig_sankey_count))
    sankey_usd_json = json.loads(pio.to_json(fig_sankey_usd))
    duration_vs_size_json = json.loads(pio.to_json(fig_duration_vs_size))
    size_dist_json = json.loads(pio.to_json(fig_size_dist))
    route_heatmap_json = json.loads(pio.to_json(fig_route_heatmap))
    token_breakdown_json = json.loads(pio.to_json(fig_token_breakdown))
    duration_dist_json = json.loads(pio.to_json(fig_duration_dist))
    activity_pattern_json = json.loads(pio.to_json(fig_activity_pattern))
    net_flows_json = json.loads(pio.to_json(fig_net_flows))

    # ========== HTML TEMPLATE ==========
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{plugin_name} - Plugin Analysis</title>
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
        .back-link {{
            color: {MUTED_TEXT};
            text-decoration: none;
            font-size: 12px;
        }}
        .back-link:hover {{
            color: {ACCENT_COLOR};
        }}
        .stats-bar {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
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
            font-size: 18px;
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
        }}
        .plugin-nav {{
            overflow-x: auto;
        }}

        @media (max-width: 1200px) {{
            .grid {{
                grid-template-columns: 1fr;
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
            .header {{
                flex-direction: column;
                align-items: flex-start;
                gap: 10px;
            }}
            .header h1 {{
                font-size: 15px;
                letter-spacing: 1px;
            }}
            .back-link {{
                font-size: 11px;
            }}
            .chart-container {{
                padding: 10px;
                margin-bottom: 10px;
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
            .back-link {{
                font-size: 10px;
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
        <h1>⬡ {plugin_name.upper()}</h1>
        <a href="../index.html" class="back-link">← Back to Dashboard</a>
    </div>

    <div class="stats-bar">
        <div class="stat">
            <div class="stat-label">Total Transfers</div>
            <div class="stat-value">{total_transfers:,}</div>
        </div>
        <div class="stat">
            <div class="stat-label">Total Volume</div>
            <div class="stat-value">${total_volume_usd/1e6:.2f}M</div>
            <div class="stat-sub">{len(plugin_with_usd):,} with USD data</div>
        </div>
        <div class="stat">
            <div class="stat-label">Avg Transfer</div>
            <div class="stat-value">${avg_transfer_usd:,.0f}</div>
            <div class="stat-sub">Median: ${median_transfer_usd:,.0f}</div>
        </div>
        <div class="stat">
            <div class="stat-label">Avg Duration</div>
            <div class="stat-value">{avg_duration:.1f}s</div>
            <div class="stat-sub">p95: {p95_duration:.1f}s</div>
        </div>
        <div class="stat">
            <div class="stat-label">Top Route</div>
            <div class="stat-value">{top_route[0]} → {top_route[1]}</div>
            <div class="stat-sub">{top_route_count:,} transfers</div>
        </div>
        <div class="stat">
            <div class="stat-label">Top Token</div>
            <div class="stat-value">{top_token_symbol}</div>
            <div class="stat-sub">{top_token_count:,} transfers</div>
        </div>
    </div>

    <div class="grid">
        <div class="chart-container">
            <div id="chart-sankey-count"></div>
        </div>

        <div class="chart-container">
            <div id="chart-sankey-usd"></div>
        </div>

        <div class="chart-container full-width">
            <div id="chart-duration-vs-size"></div>
        </div>

        <div class="chart-container">
            <div id="chart-size-dist"></div>
        </div>

        <div class="chart-container">
            <div id="chart-duration-dist"></div>
        </div>

        <div class="chart-container">
            <div id="chart-net-flows"></div>
        </div>

        <div class="chart-container">
            <div id="chart-token-breakdown"></div>
        </div>

        <div class="chart-container">
            <div id="chart-activity-heatmap"></div>
        </div>

        <div class="chart-container">
            <div id="chart-route-heatmap"></div>
        </div>
    </div>

    <div class="plugin-nav" style="margin-top: 30px; padding: 20px; background: {CARD_BG}; border: 1px solid {GRID_COLOR};">
        <h2 style="color: {ACCENT_COLOR}; font-size: 14px; margin-bottom: 15px; letter-spacing: 1px;">⬡ PLUGIN DETAIL PAGES</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px;">
"""

    # Generate plugin links
    plugin_counts = df.groupby('plugin').size().sort_values(ascending=False)
    for other_plugin in plugin_counts.index:
        is_current = other_plugin == plugin_name
        bg_color = GRID_COLOR if is_current else DARK_BG
        html_content += f"""
            <a href="{other_plugin}.html" style="display: block; padding: 12px 15px; background: {bg_color}; border: 1px solid {GRID_COLOR}; color: {TEXT_COLOR}; text-decoration: none; border-left: 2px solid {ACCENT_COLOR}; transition: all 0.2s;">
                <div style="font-size: 12px; font-weight: bold; color: {ACCENT_COLOR};">{other_plugin}</div>
                <div style="font-size: 10px; color: {MUTED_TEXT}; margin-top: 4px;">{plugin_counts[other_plugin]:,} transfers{' (current)' if is_current else ' →'}</div>
            </a>
"""

    html_content += f"""
        </div>
    </div>

    <script>
        const config = {{
            displayModeBar: false,
            responsive: true
        }};

        Plotly.newPlot('chart-sankey-count', {json.dumps(sankey_count_json['data'])}, {json.dumps(sankey_count_json['layout'])}, config);
        Plotly.newPlot('chart-sankey-usd', {json.dumps(sankey_usd_json['data'])}, {json.dumps(sankey_usd_json['layout'])}, config);
        Plotly.newPlot('chart-duration-vs-size', {json.dumps(duration_vs_size_json['data'])}, {json.dumps(duration_vs_size_json['layout'])}, config);
        Plotly.newPlot('chart-size-dist', {json.dumps(size_dist_json['data'])}, {json.dumps(size_dist_json['layout'])}, config);
        Plotly.newPlot('chart-duration-dist', {json.dumps(duration_dist_json['data'])}, {json.dumps(duration_dist_json['layout'])}, config);
        Plotly.newPlot('chart-route-heatmap', {json.dumps(route_heatmap_json['data'])}, {json.dumps(route_heatmap_json['layout'])}, config);
        Plotly.newPlot('chart-token-breakdown', {json.dumps(token_breakdown_json['data'])}, {json.dumps(token_breakdown_json['layout'])}, config);
        Plotly.newPlot('chart-activity-heatmap', {json.dumps(activity_pattern_json['data'])}, {json.dumps(activity_pattern_json['layout'])}, config);
        Plotly.newPlot('chart-net-flows', {json.dumps(net_flows_json['data'])}, {json.dumps(net_flows_json['layout'])}, config);
    </script>
</body>
</html>
"""

    # Write to file
    filename = f"plugins/{plugin_name}.html"
    with open(filename, 'w') as f:
        f.write(html_content)

    print(f"    ✓ {filename} ({total_transfers:,} transfers)")


# ========== GENERATE ALL PLUGIN PAGES ==========
print("\n" + "="*50)
print("Generating plugin detail pages...")
print("="*50)

all_plugins = df['plugin'].unique()
print(f"Found {len(all_plugins)} plugins")

for plugin in sorted(all_plugins):
    create_plugin_page(plugin, df, all_plugins)

print(f"\n✓ Generated {len(all_plugins)} plugin detail pages in plugins/")
