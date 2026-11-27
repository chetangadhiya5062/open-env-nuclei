# Troyanskaya et al. (2001): KNN imputation

- **Citation:** Troyanskaya, O., Cantor, M., Sherlock, G., Brown, P., Hastie, T., Tibshirani, R., Botstein, D., &
  Altman, R. B. (2001). Missing value estimation methods for DNA microarrays. *Bioinformatics*, 17(6), 520-525.
- **DOI:** https://doi.org/10.1093/bioinformatics/17.6.520
- **Access:** free to read on the publisher's site (Oxford Academic); no redistribution licence, so no PDF stored.

## Notes
- Compares SVD-based, KNN-based and row-average imputation; KNN imputation (weighted average of the k most similar
  rows) is robust and beats mean imputation.
- **Used in:** `featureforge/steps/cleaning.py` (`impute`, `numeric_strategy: knn` -> scikit-learn `KNNImputer`, fitted
  on at most `FF_MAX_FIT_ROWS` rows).
