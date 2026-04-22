# Explainable-Interpretable-and-Fair-AI
Comparative Study of Local Explanation Methods for Vision Transformers: LIME vs SHAP vs Counterfactual Explanations

## Emil Goryachih evaluation pipeline

The repository now includes a baseline quantitative pipeline for:

- SHAP explanations integrated with the existing ViT CIFAR-10 wrapper
- reproducible SHAP background/reference-set manifests under `artifacts/reference_sets/`
- baseline faithfulness, stability, and runtime evaluation under `artifacts/eval/`
- comparison tables and summaries under `artifacts/reports/`

Example commands:

```bash
python scripts/run_shap.py --device cpu --subset-size 5 --background-size 32
python scripts/run_faithfulness_eval.py --device cpu --subset-size 3
python scripts/run_stability_eval.py --device cpu --subset-size 3
python scripts/run_runtime_eval.py --device cpu --subset-size 3
python scripts/build_report.py
```
