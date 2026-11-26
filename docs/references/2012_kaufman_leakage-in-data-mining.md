# Kaufman, Rosset, Perlich & Stitelman (2012): leakage in data mining

- **Citation:** Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation,
  detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), Article 15.
- **DOI:** https://doi.org/10.1145/2382577.2382579
- **Access:** ACM Digital Library (subscription). No PDF stored. The openly available follow-up on the same topic,
  Kapoor & Narayanan (2023), is stored as `2023_kapoor_leakage-reproducibility-crisis.pdf`.

## Notes
- Defines leakage as information in the training data that would not be legitimately available at prediction
  time; shows real competition examples (IDs that encode the label, features recorded after the outcome).
- Recommends "learn-predict separation": every statistic must be learned only from data available before the
  prediction point, and validation must mimic deployment.
- **Used in:** `featureforge/profile.py` (`possible_target_leak`, `target_copy`, `target_name_in_column`, ID warnings),
  `featureforge/auto.py` (drops flagged leak columns), `featureforge/split.py` (duplicates/groups never straddle the
  split, time split), `featureforge/pipeline.py` (fit on train / transform on new data), `training/fe_benchmark.py`
  (pipeline fitted inside each CV fold).
