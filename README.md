# cycle_time_CatBoostRegressorand-gemma-2-9b-it-mistralai-Mistral-7B-Instruct-v0.3

I compared two approaches to predict true cycle time (sec/part) from manufacturing data.

First, I engineered the target directly from raw shop-floor logs:
I cleaned the data (removed zero-quantity rows, filtered extreme downtime, trimmed outliers), and—most importantly—prevented part-level leakage by using GroupShuffleSplit + GroupKFold. This way, the test set truly represents the real-life “new part” generalization scenario.

With tabular ML (CatBoostRegressor), the results were very strong:
RMSE ≈ 2.99 sec, sMAPE ≈ 0.89%, R² ≈ 0.9946.
This highlighted how powerful tabular models can be when you combine the right target definition, meaningful features (work_s, downtime, utilization, time features), and leakage-free validation.

In parallel, I explored an open-source LLM-based embedding approach:
I converted each row into text, aimed to generate embeddings using Mistral/Gemma (7B–9B) models, then concatenated embeddings with numeric features and trained Ridge/GBDT regressors. The idea was to use the LLM as a feature extractor, not as a direct regressor. However, in practice the LLM benchmark was limited by GPU/compute availability and (for some models) Hugging Face gated access requirements.

Overall, for this use case the most practical and best-performing solution so far is CatBoost + leakage-free CV. I plan to complete the LLM benchmark once compute and access constraints are resolved.
