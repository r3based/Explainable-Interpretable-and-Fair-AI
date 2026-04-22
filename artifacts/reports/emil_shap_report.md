# Emil Goryachih SHAP Report

## Setup
- Device: `cpu`
- Seed: `42`
- Subset size: `3` images
- Reference set: `32` images, strategy `stratified`, split `train`
- Reference manifest: `artifacts\reference_sets\cifar10_train_stratified_32_seed42.json`
- `scripts/run_shap.py` SHAP nsamples: `128`
- `scripts/run_runtime_eval.py` SHAP nsamples: `64`

## Reference Set
- airplane: 4
- automobile: 4
- bird: 3
- cat: 3
- deer: 3
- dog: 3
- frog: 3
- horse: 3
- ship: 3
- truck: 3

## SHAP Run Summary
- Mean standalone SHAP runtime from `scripts/run_shap.py`: `35.38` sec/image
- Median standalone SHAP runtime from `scripts/run_shap.py`: `35.10` sec/image
- Runtime benchmark mean from `scripts/run_runtime_eval.py`: `18.92` sec/image
- Benchmark std: `0.04` sec
- The benchmark runtime is lower because the runtime evaluation used fewer SHAP samples than the standalone artifact-generation run.

| dataset_index | true_class | predicted_class | explained_class | confidence | runtime_s | heatmap | figure |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6545 | truck | truck | truck | 0.9873 | 36.03 | `artifacts\shap\shap_00_truck_heatmap.npy` | `artifacts\shap\shap_00_truck.png` |
| 892 | horse | horse | horse | 0.9886 | 35.10 | `artifacts\shap\shap_01_horse_heatmap.npy` | `artifacts\shap\shap_01_horse.png` |
| 7738 | bird | dog | dog | 0.3543 | 35.01 | `artifacts\shap\shap_02_dog_heatmap.npy` | `artifacts\shap\shap_02_dog.png` |

## Quantitative Results
- Faithfulness deletion AUC mean: `0.2598`
- Faithfulness deletion AUC std: `0.0325`
- Faithfulness insertion AUC mean: `0.3513`
- Faithfulness insertion AUC std: `0.1396`
- Stability correlation mean: `0.3378`
- Stability correlation std: `0.1274`
- Stability top-k IoU mean: `0.1525`
- Stability top-k IoU std: `0.0631`

## Comparison Notes
- Best deletion AUC method in this baseline run: `shap`
- Best insertion AUC method in this baseline run: `lime`
- Most stable method in this baseline run: `counterfactual`
- Most computationally expensive method: `counterfactual`
- SHAP reasonable trade-off according to the generated summary: `True`

## Interpretation
- On this CPU baseline subset, SHAP achieved the best deletion AUC among the three methods.
- SHAP did not dominate insertion AUC or stability, but it was also not Pareto-dominated when quality and runtime were considered together.
- Counterfactual explanations were the most stable and the most computationally expensive in this run.
- LIME achieved the best insertion AUC in this run.
