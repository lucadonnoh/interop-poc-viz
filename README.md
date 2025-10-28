# Across Protocol Interop Transfer Dashboard

Comprehensive visualization dashboard for [Across Protocol](https://across.to/) interop transfer data, analyzing cross-chain bridge performance, network flows, and protocol metrics.

## Overview

This project provides real-time analytics for cross-chain transfers across 15+ protocols and 4 major chains (Arbitrum, Base, Ethereum, Optimism). The dashboard offers detailed insights into transfer volumes, speeds, routes, and token flows.

**Current Data:** 103,020 transfers | $65.4M total volume | 15 protocols | 4 chains

## Features

### Main Dashboard (`index.html`)
- **Network Flow Sankey Diagrams** - Visualize transfer count and USD volume flows between chains with token-level breakdown
- **Route Speed Analysis** - Compare top 3 fastest protocols per route with avg/p95/p99 latency metrics
- **Transfer Timelines** - Hourly transfer count and USD volume trends
- **Net USD Flows** - Chain-level incoming vs outgoing volume analysis
- **Protocol Performance** - Size specialization analysis (small/medium/large transfers)
- **Token Market Overview** - Transfer count and volume distribution by token
- **Chain Token Flows** - Per-chain token activity visualization
- **Transfer Value Distribution** - Log-scale distribution with median/mean/total stats
- **Missing Data Analysis** - Data completeness tracking by protocol

### Plugin-Specific Pages (`plugins/*.html`)
Each of the 15 protocols has a dedicated analysis page with:
- **Summary Statistics** - Total transfers, volume, average transfer size, duration metrics
- **Network Flow Sankey Diagrams** - Transfer count and USD volume flows (with token breakdown)
- **Duration vs Size Scatter** - Relationship between transfer value and completion time
- **Size & Duration Distributions** - Log-scale histograms
- **Route Heatmap** - Transfer counts per source-destination pair
- **Token Breakdown** - Pie charts showing token distribution (count & volume)
- **Activity Patterns** - Day of week & hour of day heatmap
- **Net USD Flows** - Per-chain net flow analysis
- **Plugin Navigation** - Direct links to all other plugin pages

## Data Source

Data is fetched from a PostgreSQL database containing live interop transfer events:

### Refresh Data

To update with the latest transfers:

```bash
# Export fresh data from database
psql "$DATABASE_URL" -c "COPY (
  SELECT
    plugin,
    \"transferId\",
    type,
    timestamp,
    duration,
    \"srcChain\",
    \"dstChain\",
    \"srcAbstractTokenId\",
    \"srcTokenAddress\",
    \"srcAmount\",
    \"srcValueUsd\",
    \"dstValueUsd\",
    \"isProcessed\"
  FROM \"InteropTransfer\"
  ORDER BY timestamp ASC
) TO STDOUT WITH CSV HEADER" > interop_fresh.csv

# Regenerate all visualizations
python create_dashboard.py
python create_plugin_pages.py
```

**Note:** Ensure `DATABASE_URL` environment variable is set with your PostgreSQL connection string.

## Technical Stack

- **Python** - Data processing and chart generation
  - `pandas` - Data manipulation and aggregation
  - `plotly` - Interactive visualizations
- **Plotly.js** - Client-side chart rendering
- **PostgreSQL** - Data source
- **HTML/CSS** - Dashboard UI with dark theme

## File Structure

```
.
├── create_dashboard.py           # Main dashboard generator
├── create_plugin_pages.py        # Plugin-specific page generator
├── interop_fresh.csv              # Current transfer data (auto-updated)
├── index.html                     # Main dashboard (generated)
├── plugins/                       # Plugin detail pages (generated)
│   ├── across.html
│   ├── relay.html
│   └── ...
├── token_symbol_cache.json        # Token metadata cache
└── venv/                          # Python virtual environment
```

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install pandas plotly psycopg2-binary
```

## Usage

### Generate All Visualizations

```bash
# Main dashboard
python create_dashboard.py

# Plugin-specific pages
python create_plugin_pages.py
```

### View Locally

Simply open `index.html` in a web browser. All charts are embedded and interactive.

## Key Metrics Explained

- **Average Duration** - Mean transfer completion time
- **p95/p99 Duration** - 95th and 99th percentile latency (worst-case scenarios)
- **Net USD Flows** - Incoming volume minus outgoing volume per chain
- **Token Breakdown** - Distribution filtered to tokens with ≥0.1% share (others grouped as "Other")

## Supported Protocols

1. **relay** - 61,269 transfers
2. **relay-simple** - 23,573 transfers
3. **across** - 13,117 transfers
4. **stargate** - 1,574 transfers
5. **layerzero-v2-ofts** - 1,110 transfers
6. **debridge-dln** - 1,094 transfers
7. **hyperlane-hwr** - 535 transfers
8. **oneinch-fusion-plus** - 183 transfers
9. **squid-coral** - 147 transfers
10. **opstack-standardbridge** - 140 transfers
11. **axelar** - 129 transfers
12. **hyperlane-merkly-tokenbridge** - 50 transfers
13. **circle-gateway** - 45 transfers
14. **ccip** - 35 transfers
15. **celer** - 19 transfers

## Features by Persona

### Protocol Teams
- Compare your protocol's speed against competitors per route
- Analyze token support and volume distribution
- Identify performance outliers (p95/p99 metrics)

### Users with Transfer Issues
- Check typical duration ranges for your route
- See which protocols are fastest for specific routes via route speed analysis
- View network flow patterns with Sankey diagrams

### Developers/Integrators
- Route and network flow analysis for integration prioritization
- Token flow patterns across chains
- Data completeness tracking

## Contributing

This is a visualization project for Across Protocol interop data. For updates or improvements:

1. Modify the generator scripts (`create_dashboard.py`, `create_plugin_pages.py`)
2. Test locally
3. Commit changes

## Data Freshness

The dashboard displays the most recently exported data from the database. Data is exported on-demand via the refresh command above.

**Last Update:** Check the timeline charts for the latest timestamp range.

## License

MIT
