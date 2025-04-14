#!/usr/bin/env python3
"""
This script loads performance data from a JSON file and creates a separate
grouped bar chart for each unique "request_rate" value (representing Questions Per/Sec, QPS).
Each chart compares the metrics "mean_ttft_ms", "mean_tpot_ms", and "mean_itl_ms" between the
two frameworks ("sgl" and "vllm").

Prerequisites:
    pip install pandas plotly
    (Optional) pip install kaleido for PNG export.
"""

import json
import os
import argparse
import math
import pandas as pd
import plotly.express as px

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

def plot_metrics_for_qps(df, qps, model_name, export_png=False, export_html=False):
    # Filter the data for the provided request_rate (QPS) value.
    df_qps = df[df['request_rate'] == qps]
    if df_qps.empty:
        print(f"No data found for request_rate (QPS) = {qps}")
        return

    # Define the metrics and compute the mean per framework.
    metrics = ["mean_ttft_ms", "mean_tpot_ms", "mean_itl_ms"]
    df_summary = df_qps.groupby("framework")[metrics].mean()

    # Transpose for easier plotting and reset index.
    df_summary = df_summary.transpose().reset_index().rename(columns={"index": "metric"})

    # Melt the DataFrame into a tidy format.
    df_melt = pd.melt(df_summary, id_vars="metric", var_name="framework", value_name="value")

    # Determine the QPS label: if qps is numeric infinity or the string "inf", show "infinite".
    if (isinstance(qps, (int, float)) and math.isinf(qps)) or (isinstance(qps, str) and qps.lower() == "inf"):
        qps_label = "infinite"
    else:
        qps_label = str(qps)

    # Create a grouped bar chart with a title that includes the model name and QPS label.
    title = f"Model: {model_name} | QPS: {qps_label}"
    fig = px.bar(
        df_melt,
        x="metric",
        y="value",
        color="framework",
        barmode="group",
        title=title,
        labels={"value": "ms", "metric": "Metric"},
        # Set specific colors for each framework.
        color_discrete_map={"vllm": "#f17322", "sgl": "#f8c518"},
        # Specify the order so "vllm" is plotted first.
        category_orders={"framework": ["vllm", "sgl"]}
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(title=f"Questions Per/Sec (QPS: {qps_label})"),
        yaxis=dict(showgrid=True, gridcolor="#edebf0")
    )

    if export_png:
        png_filename = f"bar_chart_qps_{qps_label}.png"
        try:
            fig.write_image(png_filename)
            print(f"Figure for QPS = {qps_label} exported to {png_filename}")
        except Exception as e:
            print(f"Failed to export figure for QPS = {qps_label} to PNG. Error: {e}")
    if export_html:
        html_filename = f"bar_chart_qps_{qps_label}.html"
        try:
            # Generate the Plotly HTML as a string.
            html_content = fig.to_html(include_plotlyjs="cdn")
            # Wrap the HTML content in raw tags so Jekyll doesn't process its curly braces or else it fails to build :-(
            wrapped_html = "{% raw %}\n" + html_content + "\n{% endraw %}"
            with open(html_filename, 'w') as f:
                f.write(wrapped_html)
            print(f"Figure for QPS = {qps_label} exported to {html_filename}")
        except Exception as e:
            print(f"Failed to export figure for QPS = {qps_label} to HTML. Error: {e}")

    # If no export format is specified, show the figure in the browser.
    if not (export_png or export_html):
        fig.show()

def main():
    parser = argparse.ArgumentParser(
        description="Generate grouped bar charts for each unique request_rate (QPS) value."
    )
    parser.add_argument(
        "--export",
        metavar="FORMAT",
        choices=["png", "html"],
        help='Export the charts as the specified format ("png" or "html"). If not provided, charts are shown in a browser.'
    )
    parser.add_argument(
        "--model",
        metavar="MODEL",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name to include in the chart title. Default is 'meta-llama/Llama-3.1-8B-Instruct'."
    )
    args = parser.parse_args()

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

    unique_qps = df['request_rate'].unique()
    print(f"Found the following request_rate (QPS) values: {unique_qps}")

    for qps in unique_qps:
        plot_metrics_for_qps(df, qps, model_name=args.model, export_png=export_png, export_html=export_html)

if __name__ == "__main__":
    main()
