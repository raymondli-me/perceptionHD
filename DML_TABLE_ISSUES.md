# DML Table Issues

## Current Status

The DML table visualization has been updated to be generic (X/Y instead of AI/SC), but there are some important issues with the values:

### ✅ Fixed Issues
1. **PC List**: Now correctly shows dynamically selected PCs (e.g., PC0, PC2, PC5, PC1, PC25)
2. **Labels**: All AI/SC labels replaced with generic X/Y throughout
3. **Color by menu**: Shows "X/Y" instead of "AI/SC"

### ❌ Remaining Issues

#### 1. Missing Crossfitted R² Values
The DML table has two columns:
- **Left column**: "Non Cross-Fitted" 
- **Right column**: "Cross-Fitted (5-fold)"

Currently, the pipeline only computes non-crossfitted R² values:
- `top5_r2_x`: 0.127 (non-crossfitted)
- `top5_r2_y`: 0.526 (non-crossfitted)
- `all_r2_x`: 0.524 (non-crossfitted)
- `all_r2_y`: 0.612 (non-crossfitted)

The visualization code incorrectly puts the same non-crossfitted value in both columns because crossfitted R² values are not computed by the pipeline.

**Expected behavior**: The right column should show crossfitted R² values, which are typically lower than non-crossfitted values due to prevention of overfitting.

#### 2. Missing Standard Error for Naive Model
The DML results include:
- `se_200`: 0.0078 (standard error for 200 PC model)
- `se_top5`: 0.0078 (standard error for top 5 PC model)
- `se_naive`: MISSING

The naive model's standard error is not computed by the pipeline.

#### 3. Hardcoded Values Still Present
Some values in the HTML appear to be hardcoded from the original template:
- 200 PC model R² values: 0.919 (Y), 0.814 (X) for non-crossfitted
- 200 PC model R² values: 0.505 (Y), -0.023 (X) for crossfitted

These should be replaced with actual computed values from the analysis.

## Recommendations

To fully fix the DML table:

1. **Compute crossfitted R² values**: Modify the pipeline to use cross-validation when computing R² scores for both X and Y predictions
2. **Compute naive standard error**: Add calculation of standard error for the naive model
3. **Separate storage**: Store both crossfitted and non-crossfitted R² values in the DML results
4. **Update visualization**: Ensure the visualization correctly maps crossfitted values to the right column

## Current Workaround

The table currently shows the same (non-crossfitted) R² values in both columns, which is misleading. Users should be aware that the "Cross-Fitted" column values are actually non-crossfitted.