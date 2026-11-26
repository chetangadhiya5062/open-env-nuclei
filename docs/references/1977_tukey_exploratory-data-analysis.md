# Tukey (1977): Exploratory Data Analysis (IQR fences)

- **Citation:** Tukey, J. W. (1977). *Exploratory Data Analysis.* Addison-Wesley. ISBN 978-0201076165.
- **Access:** book (copyrighted). No copy stored.

## Notes
- Box-plot fences: values below Q1 - k*IQR or above Q3 + k*IQR are "outside" (k = 1.5) or "far out" (k = 3).
- **Used in:** `featureforge/profile.py` (1.5*IQR outlier counts and the `outliers` warning),
  `featureforge/steps/cleaning.py` (`outliers`, `method: iqr`, parameter `k`).
