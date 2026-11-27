# Siddiqi (2006): Credit Risk Scorecards (Population Stability Index)

- **Citation:** Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring.*
  Wiley. ISBN 978-0471754510.
- **Access:** book (copyrighted). No copy stored.

## Notes
- Population Stability Index `PSI = sum((cur% - ref%) * ln(cur% / ref%))` over bins of the reference distribution;
  rule of thumb: < 0.1 no significant shift, 0.1-0.25 moderate, > 0.25 major shift.
- **Used in:** `featureforge/profile.py` (`drift_report`: PSI over reference deciles / top categories, same
  thresholds), shown in the dashboard's Apply tab and in run summaries when a split is used.
