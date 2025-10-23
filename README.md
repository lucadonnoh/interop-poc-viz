# Across Interop Visualization

Interactive visualization of [Across Protocol](https://across.to/) interop transfer data, showing the relationship between USD value and duration across different token types.

## Features

- **Interactive scatter plot** with log-log scale for both axes
- **Dynamic filtering** - click legend items to show/hide specific tokens
- **Hover details** - see transfer ID, plugin, USD value, and duration
- **Zoom and pan** capabilities for detailed exploration
- **Smart opacity** - automatically adjusts based on data density

## View the Visualization

Visit the live site: [GitHub Pages URL will be here after deployment]

## Data

The visualization displays 9,431 valid Across Protocol interop transfers across 10 different token types:
- USDC (circle) - 2,565 transfers
- ETH (ethereum) - 6,469 transfers
- USDT (tether) - 330 transfers
- And more...

## How to Use

1. **Toggle tokens**: Click any legend item to show/hide that token's data points
2. **Isolate a token**: Double-click a legend item to show only that token
3. **Zoom**: Click and drag to select an area to zoom into
4. **Pan**: Hold Shift and drag to pan around
5. **Reset**: Use the home icon in the toolbar to reset the view
6. **Hover**: Move your mouse over points to see detailed information

## Updating the Data

To update the visualization with new data:

1. Replace `interop.csv` with your new CSV file
2. Commit and push to the main branch:
   ```bash
   git add interop.csv
   git commit -m "Update interop data"
   git push
   ```
3. GitHub Actions will automatically regenerate the visualization
4. The updated site will be live in a few minutes

You can also manually regenerate the visualization locally:
```bash
python generate_visualization.py
```

## Technical Details

Built with:
- Python (pandas, plotly)
- Plotly.js for interactive visualization
- GitHub Pages for hosting
- GitHub Actions for automatic updates

## Data Insights

- Most transfers cluster in the low-value, short-duration range
- Clear horizontal banding at specific duration values (2, 3, 4, etc.)
- Some outliers show very high durations (7000+) regardless of value
- High-value transfers (>$1M) tend to have moderate durations
