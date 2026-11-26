# Iglewicz & Hoaglin (1993): modified z-score

- **Citation:** Iglewicz, B., & Hoaglin, D. C. (1993). *How to Detect and Handle Outliers.* ASQC Basic References in
  Quality Control, Vol. 16. ASQC Quality Press.
- **Access:** book (copyrighted). No copy stored.

## Notes
- Modified z-score `M = 0.6745 * (x - median) / MAD`; values with |M| > 3.5 are potential outliers. Robust because
  median and MAD are not pulled by the outliers themselves.
- **Used in:** `featureforge/steps/cleaning.py` (`outliers`, `method: mad`, default `threshold: 3.5`).
