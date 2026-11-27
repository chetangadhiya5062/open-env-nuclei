# Micci-Barreca (2001): target encoding of high-cardinality categories

- **Citation:** Micci-Barreca, D. (2001). A Preprocessing Scheme for High-Cardinality Categorical Attributes in
  Classification and Prediction Problems. *ACM SIGKDD Explorations Newsletter*, 3(1), 27-32.
- **DOI:** https://doi.org/10.1145/507533.507538
- **Access:** ACM Digital Library (subscription / pay-per-view). No PDF stored: not openly licensed.

## Notes
- Replaces a category by a blend of its own target statistic and the prior (overall) statistic; the blend weight
  grows with the category count, so rare categories shrink toward the prior (empirical-Bayes style smoothing).
- Works for binary, multi-class (one statistic per class) and continuous targets.
- **Used in:** `featureforge/steps/encoding.py` (`target_encode`: `(n * mean + m * prior) / (n + m)`, `smoothing = m`,
  one column per class for multi-class). Leakage protection (out-of-fold / ordered) comes from the CatBoost paper.
