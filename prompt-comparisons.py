#!/usr/bin/env python3
"""
This script loads performance data from a JSON file and generates summary figures
for each metric ("mean_ttft_ms", "mean_tpot_ms", and "mean_itl_ms").
Each summary figure includes a bar chart and an accompanying table,
and the title displays both the abbreviated metric and its full name in parentheses.

It supports exporting the figures as PNG or HTML files using the flag:
    --export FORMAT

Where FORMAT is either "png" or "html". If the flag is not provided, the figures
are displayed interactively in a browser.

Prerequisites:
    pip install pandas plotly
    (Optional) pip install kaleido for PNG export.
"""

import json
import os
import argparse
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_data(file_name):
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    data.append(record)
                except Exception as e:
                    print("Error parsing line:", line)
                    print(e)
    return pd.DataFrame(data)

def compute_winner(v, s):
    """
    Returns a tuple (Fastest, percent_faster) with percent_faster rounded to two decimals.
    """
    if pd.isna(v) or pd.isna(s):
        return ("N/A", 0)
    if v < s:
        percent_faster = 100 * (s - v) / s
        return ("vllm", round(percent_faster, 2))
    elif s < v:
        percent_faster = 100 * (v - s) / v
        return ("sgl", round(percent_faster, 2))
    else:
        return ("Tie", 0)

def plot_metric_summary_with_table(df, metric, export_png=False, export_html=False):
    # Mapping from metric abbreviation to full descriptive name.
    metric_names = {
        "mean_ttft_ms": "Mean Time To First Token",
        "mean_tpot_ms": "Mean Time Per Output Token",
        "mean_itl_ms": "Mean Inference Token Latency"
    }
    full_name = metric_names.get(metric, "")

    # Group data by num_prompts and framework; compute the mean for the metric.
    df_grouped = df.groupby(['num_prompts', 'framework'])[metric].mean().reset_index()
    # Pivot so that rows are num_prompts with columns for each framework.
    df_pivot = df_grouped.pivot(index='num_prompts', columns='framework', values=metric)
    df_pivot = df_pivot.sort_index()

    # Compute the Fastest framework and percentage faster for each num_prompts.
    results = df_pivot.apply(lambda row: compute_winner(row.get("vllm"), row.get("sgl")), axis=1)
    df_pivot['Fastest'] = results.apply(lambda x: x[0])
    df_pivot['Faster by (%)'] = results.apply(lambda x: x[1])

    # Create a subplot with two rows: the bar chart (row 1) and the table (row 2).
    fig = make_subplots(rows=2, cols=1,
                        row_heights=[0.65, 0.35],
                        shared_xaxes=True,
                        vertical_spacing=0.2,
                        specs=[[{"type": "xy"}],
                               [{"type": "table"}]])

    # Add grouped bar chart traces.
    fig.add_trace(go.Bar(
        x=df_pivot.index,
        y=df_pivot["vllm"],
        name="vllm",
        marker_color="#F17322",
        hovertemplate="num_prompts: %{x}<br>vllm: %{y:.2f} ms<extra></extra>"
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df_pivot.index,
        y=df_pivot["sgl"],
        name="sgl",
        marker_color="#f8c518",
        hovertemplate="num_prompts: %{x}<br>sgl: %{y:.2f} ms<extra></extra>"
    ), row=1, col=1)

    fig.update_yaxes(title_text=f"{metric} (ms)", row=1, col=1, showgrid=True, gridcolor="#edebf0")
    fig.update_xaxes(title_text="Number of Prompts", row=1, col=1, showgrid=True, gridcolor="#edebf0", title_standoff=25)
    fig.update_layout(barmode="group",
                      plot_bgcolor="white",
                      paper_bgcolor="white",
                      title_text=f"Summary for {metric} ({full_name})",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # Prepare table data.
    num_prompts_col = list(df_pivot.index)
    vllm_col = [f"{val:.2f}" for val in df_pivot["vllm"]]
    sgl_col = [f"{val:.2f}" for val in df_pivot["sgl"]]
    Fastest_col = df_pivot["Fastest"].tolist()
    faster_col = [f"{val:.2f}%" for val in df_pivot["Faster by (%)"]]

    # Create the table trace with updated grid line and header colors.
    table_header = dict(
        values=["num_prompts", "vllm (ms)", "sgl (ms)", "Fastest", "Faster by (%)"],
        fill_color="#FFD681",
        align="center",
        font=dict(color="#303030", size=12),
        line=dict(color="#edebf0")
    )
    table_cells = dict(
        values=[num_prompts_col, vllm_col, sgl_col, Fastest_col, faster_col],
        fill_color="white",
        align="center",
        font=dict(color="#303030", size=12),
        line=dict(color="#edebf0")
    )

    fig.add_trace(go.Table(header=table_header, cells=table_cells), row=2, col=1)

    # Increase overall figure height to show the entire table.
    fig.update_layout(height=800)

    # Export or display the figure based on provided flags.
    if export_png:
        png_filename = f"summary_{metric}.png"
        try:
            fig.write_image(png_filename)
            print(f"Figure for {metric} exported to {png_filename}")
        except Exception as e:
            print(f"Failed to export figure for {metric} to PNG. Error: {e}")

    if export_html:
        html_filename = f"summary_{metric}.html"
        try:
            # Generate Plotly HTML content as a string.
            html_content = fig.to_html(include_plotlyjs="cdn")
            # Wrap the content in raw tags to prevent Jekyll from processing the {{ }}.
            wrapped_html = "{% raw %}\n" + html_content + "\n{% endraw %}"
            with open(html_filename, 'w') as f:
                f.write(wrapped_html)
            print(f"Figure for {metric} exported to {html_filename}")
        except Exception as e:
            print(f"Failed to export figure for {metric} to HTML. Error: {e}")

    if not (export_png or export_html):
        fig.show()

def main():
    parser = argparse.ArgumentParser(description="Generate performance summary figures.")
    parser.add_argument(
        "--export",
        metavar="FORMAT",
        choices=["png", "html"],
        help='Export the figures as the specified format ("png" or "html"). If not provided, figures are shown in a browser.'
    )
    args = parser.parse_args()

    # Set export flags based on the provided argument.
    export_png = False
    export_html = False
    if args.export:
        if args.export.lower() == "png":
            export_png = True
        elif args.export.lower() == "html":
            export_html = True

    file_path = "results.json"
    if not os.path.exists(file_path):
        print("File 'results.json' not found. Please ensure it exists in the current directory.")
        return

    df = load_data(file_path)
    if df.empty:
        print("No data loaded. Please check your 'results.json' file.")
        return

    print("Data loaded successfully:")
    print(df.head())

    # For each metric, generate the summary figure with the table below.
    metrics_to_summary = ["mean_ttft_ms", "mean_tpot_ms", "mean_itl_ms"]
    for metric in metrics_to_summary:
        plot_metric_summary_with_table(df, metric,
                                       export_png=export_png,
                                       export_html=export_html)

if __name__ == "__main__":
    main()
