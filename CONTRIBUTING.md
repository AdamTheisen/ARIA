# Contributing

Use Python 3.13 when practical. Keep adapters, processing, cubes, workflows, and dashboard presentation separated.

```bash
python -m pip install -e .
streamlit run aria/dashboard/app.py
```

Optional ADAPT support: `conda install conda-forge::arm-adapt`. Document which live network/service tests were actually run.


## AI-assisted contributions

AI-assisted code and documentation are allowed, but contributors should
disclose material use of generative AI in pull requests when it affects
implementation or scientific logic. Generated code remains subject to
the same review, testing, provenance, and scientific-validation
requirements as manually written code.
