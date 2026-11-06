# Limitations

1. **Development exposure.** Each of the six landscapes was held out once, but all six later influenced development, interpretation or selection. This is not untouched external validation.
2. **Only six independent landscapes.** Hundreds of thousands of terrain units do not create hundreds of thousands of independent geographical-transfer replicates.
3. **Operational labels.** Unlabelled units are not confirmed stable terrain. Inventories differ in geometry, completeness, observation support, trigger timing and minimum mapping unit.
4. **Prevalence dependence.** AP changes with the positive-label fraction. PR lift normalises by prevalence but does not remove every inventory-definition effect.
5. **Ranking versus probability.** AP, lift and ROC-AUC assess ranking. Calibration transferred poorly, so raw scores should not be treated as probabilities.
6. **Uncertainty scope.** Spatial-block intervals describe within-landscape spatial sensitivity conditional on the six fitted models and inventories; they do not estimate variation over future landscapes.
7. **Post-selection analyses.** Elevation, training-composition, factor and some inventory analyses were exploratory or conducted after the primary model comparison.
8. **Model grids.** Hyperparameter grids were finite and do not prove that the selected family is universally best.
9. **No causal inference.** Feature coefficients, SHAP values and correlations describe model behaviour; they do not establish physical causation.
10. **Geological history.** Only a partial Kumamoto reconstruction passed the frozen gates. History-inclusive predictive benefit was not tested in a completed V2 model.
11. **Data-release gap.** Full source and derived datasets are not included because redistribution rights are not cleared across all inputs.
12. **No seventh-landscape result.** Candidate/lockbox infrastructure exists in the historical source snapshot, but no completed untouched external-validation result is claimed.
