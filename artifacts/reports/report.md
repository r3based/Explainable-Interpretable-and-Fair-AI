# Evaluation Report

## Table A — Explanation Quality

| method | num_images | deletion_auc_mean | deletion_auc_std | insertion_auc_mean | insertion_auc_std | stability_corr_mean | stability_iou_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| counterfactual | 3 | 0.3615860641002655 | 0.1004248708486557 | 0.3535864055156708 | 0.13334086537361145 | 0.6293869018554688 | 0.6246623992919922 |
| lime | 3 | 0.31546029448509216 | 0.12868084013462067 | 0.4891629219055176 | 0.16415588557720184 | 0.14142508804798126 | 0.0709167867898941 |
| shap | 3 | 0.25977596640586853 | 0.03250419348478317 | 0.35127124190330505 | 0.1396356225013733 | 0.33784613013267517 | 0.1525244116783142 |

## Table B — Computational Cost

| method | device | runtime_mean_sec | runtime_std_sec | runtime_median_sec | peak_memory_mb |
| --- | --- | --- | --- | --- | --- |
| counterfactual | cpu | 24.499695499997568 | 0.07304996134009632 | 24.49767189999693 | 0.0 |
| lime | cpu | 20.536443100000422 | 0.05195875531116508 | 20.50854249999975 | 0.0 |
| shap | cpu | 18.920383500000753 | 0.04095761730807157 | 18.910219799996412 | 0.0 |

## Summary

No single method dominated both faithfulness metrics; best deletion AUC: shap, best insertion AUC: lime.
counterfactual dominated the baseline stability metrics.
Most computationally expensive method by mean runtime: counterfactual.
SHAP was not Pareto-dominated under the baseline quality/cost metrics.