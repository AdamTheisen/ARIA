# Contributing

Use Python 3.13 when practical. Keep adapters, processing, cubes, workflows, and dashboard presentation separated.

```bash
python -m pip install -e .
streamlit run aria/dashboard/app.py
```

Optional ADAPT support: `conda install conda-forge::arm-adapt`. Document which live network/service tests were actually run.
