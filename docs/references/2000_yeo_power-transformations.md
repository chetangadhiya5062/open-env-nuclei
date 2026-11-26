# Yeo & Johnson (2000): power transformations

- **Citation:** Yeo, I.-K., & Johnson, R. A. (2000). A new family of power transformations to improve normality or
  symmetry. *Biometrika*, 87(4), 954-959.
- **DOI:** https://doi.org/10.1093/biomet/87.4.954
- **Access:** Oxford University Press / JSTOR (subscription). No PDF stored.

## Notes
- Extends Box-Cox to zero and negative values with one parameter lambda estimated by maximum likelihood.
- **Used in:** `featureforge/steps/features.py` (`numeric_transform`, `method: yeo_johnson`, via scikit-learn's
  `PowerTransformer(method="yeo-johnson", standardize=False)`; chosen automatically for skewed columns with
  negative values).
