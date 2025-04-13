# VLLM Benchmark Graphing

### Run

With `results.json` in the same directory (or use the included sample) run:

```commandline
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install pandas plotly kaleido

python prompt-comparisons.py --export html
python metric-perfs.py --export html
```

For static png graphs run:

```commandline
python prompt-comparisons.py --export png
python metric-perfs.py --export png
```
