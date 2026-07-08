# Reflection

**Which fault types were hardest to catch, and why?**
The hardest faults to catch were the subtle shifts in embedding drift (`embedding_batch` and `feature_materialization` mean shifts) that lay close to the boundaries of standard clean-data variance. Z-score and other adaptive heuristics initially seemed promising, but the baseline variance in clean cycles was high enough that they quickly generated false positives, leading to score penalties.

**What would you change about your cost/coverage tradeoff, if you had another pass?**
Given another pass, I would introduce a dynamic budget allocator that estimates resource consumption mid-stream. If the remaining budget is high, it could run slightly more sensitive checks or call more expensive tools on high-risk events (like contracts or schemas). Currently, a conservative static threshold policy is used to guarantee that FPR remains exactly 0% while staying strictly below the budget limit.
