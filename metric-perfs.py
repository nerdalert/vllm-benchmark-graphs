#!/usr/bin/env python3
"""
This script loads performance data from a JSON file and creates a separate
grouped bar chart for each unique "num_prompts" value. Each chart compares
the metrics "mean_ttft_ms", "mean_tpot_ms", and "mean_itl_ms" between the
two frameworks ("sgl" and "vllm").

It supports exporting the charts as PNG or HTML files by using the flag:
    --export FORMAT

Where FORMAT is either "png" or "html". If the flag is not provided, the
chart is displayed in a browser.

Prerequisites:
    pip install pandas plotly
    (Optional) pip install kaleido for PNG export.
"""

import json
import os
import argparse
import pandas as pd
import plotly.express as px

def load_data(file_name):
    """
    Load JSON lines from the file and return a pandas DataFrame.
    Each line in the file should be a valid JSON object.
    """
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

def plot_metrics_for_prompts(df, num_prompts, export_png=False, export_html=False):
    """
    For a given num_prompts value, create a grouped bar chart for the metrics:
    "mean_ttft_ms", "mean_tpot_ms", and "mean_itl_ms" comparing the two frameworks.
    The chart title includes both the number of prompts and the model_id(s).

    Depending on the export flags provided, the chart is either exported as PNG/HTML
    or displayed interactively in a browser.
    """
    # Filter the data for the provided num_prompts value.
    df_prompt = df[df['num_prompts'] == num_prompts]
    if df_prompt.empty:
        print(f"No data found for num_prompts = {num_prompts}")
        return

    # Extract the unique model_id(s) from this subset.
    model_ids = df_prompt['model_id'].unique()
    model_id_str = model_ids[0] if len(model_ids) == 1 else ", ".join(model_ids)

    # Define the metrics and compute the mean per framework.
    metrics = ["mean_ttft_ms", "mean_tpot_ms", "mean_itl_ms"]
    df_summary = df_prompt.groupby("framework")[metrics].mean()

    # Transpose for easier plotting and reset index.
    df_summary = df_summary.transpose().reset_index().rename(columns={"index": "metric"})

    # Melt the DataFrame into a tidy format.
    df_melt = pd.melt(df_summary, id_vars="metric", var_name="framework", value_name="value")

    # Create a grouped bar chart with a title including the model_id(s) and num_prompts.
    title = f"Model: {model_id_str} | Prompts: {num_prompts}"
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

    # Update layout with a white background and grid lines.
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#edebf0"),
        yaxis=dict(showgrid=True, gridcolor="#edebf0")
    )

    # Export the figure as PNG or HTML based on the provided export flag.
    if export_png:
        png_filename = f"bar_chart_prompts_{num_prompts}.png"
        try:
            fig.write_image(png_filename)
            print(f"Figure for num_prompts = {num_prompts} exported to {png_filename}")
        except Exception as e:
            print(f"Failed to export figure for num_prompts = {num_prompts} to PNG. Error: {e}")
    if export_html:
        html_filename = f"bar_chart_prompts_{num_prompts}.html"
        try:
            # Generate the Plotly HTML as a string.
            html_content = fig.to_html(include_plotlyjs="cdn")
            # Wrap the HTML content in raw tags so Jekyll doesn't process its curly braces.
            wrapped_html = "{% raw %}\n" + html_content + "\n{% endraw %}"
            with open(html_filename, 'w') as f:
                f.write(wrapped_html)
            print(f"Figure for num_prompts = {num_prompts} exported to {html_filename}")
        except Exception as e:
            print(f"Failed to export figure for num_prompts = {num_prompts} to HTML. Error: {e}")

    # If no export format is specified, show the figure in the browser.
    if not (export_png or export_html):
        fig.show()

def main():
    parser = argparse.ArgumentParser(
        description="Generate grouped bar charts for each unique num_prompts value."
    )
    parser.add_argument(
        "--export",
        metavar="FORMAT",
        choices=["png", "html"],
        help='Export the charts as the specified format ("png" or "html"). If not provided, charts are shown in a browser.'
    )
    args = parser.parse_args()

    # Set export flags based on the argument.
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

    # Generate a chart for each unique num_prompts value.
    unique_prompts = df['num_prompts'].unique()
    print(f"Found the following num_prompts values: {unique_prompts}")

    for prompts in unique_prompts:
        plot_metrics_for_prompts(df, prompts, export_png=export_png, export_html=export_html)

if __name__ == "__main__":
    main()
