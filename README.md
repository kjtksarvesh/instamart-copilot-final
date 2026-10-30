# Instamart Customer Intelligence & Support Copilot

## Overview

This repository implements the four-part synthetic capstone as one reproducible Python project. It cleans the supplied fictional customer/order exports into CSV and SQLite, segments customers and predicts lapse, classifies privacy-redacted support tickets, and provides a TF-IDF policy assistant plus a tool-using refund agent. All monetary values are INR. The Instamart name is used only as a scenario label; none of the data or policies is real.

The implementation uses Python, pandas, NumPy, scikit-learn, joblib and SQLite. Every model interaction uses the supplied offline `mock_llm.py`; no network call, API key, paid service or account-gated service is required.

## Reproduce

Run these commands from the repository root on Python 3.11 or newer. The scripts use fixed random seeds and regenerate the data and outputs. On Windows, activate the environment with `.venv\\Scripts\\activate` instead of the POSIX command shown below.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_data.py
python run_part_a.py
python validate.py
python run_part_b.py
python run_part_c.py
python run_part_d.py
```

`generate_data.py` and `mock_llm.py` are included at the repository root. `mock_llm.py` preserves the supplied offline interface and deterministic behaviours. `generate_data.py` prints exactly `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. The Part A validation script prints exactly `ALL CHECKPOINTS PASSED`. All generated CSVs, the SQLite database, model, JSONL logs, metrics and state checkpoints are written beneath `data/`, `outputs/` and `state/`.

## Part A

### Raw data audit

The raw shapes and inferred pandas dtypes are:

| Table | Shape | Column : dtype |
|---|---:|---|
| customers | 1,200 × 5 | customer_id object; city object; signup_date object; age float64; gender object |
| products | 40 × 4 | product_id object; product_name object; category object; price_inr int64 |
| orders | 13,084 × 9 | order_id object; customer_id object; product_id object; quantity int64; order_ts object; status object; discount_pct int64; order_amount object; delivery_minutes float64 |

Missing-value audit (count and percentage of rows in the table):

| Table | Column | Missing | Percentage |
|---|---|---:|---:|
| customers | customer_id | 0 | 0.00% |
| customers | city | 0 | 0.00% |
| customers | signup_date | 0 | 0.00% |
| customers | age | 61 | 5.08% |
| customers | gender | 30 | 2.50% |
| products | product_id | 0 | 0.00% |
| products | product_name | 0 | 0.00% |
| products | category | 0 | 0.00% |
| products | price_inr | 0 | 0.00% |
| orders | order_id | 0 | 0.00% |
| orders | customer_id | 0 | 0.00% |
| orders | product_id | 0 | 0.00% |
| orders | quantity | 0 | 0.00% |
| orders | order_ts | 0 | 0.00% |
| orders | status | 0 | 0.00% |
| orders | discount_pct | 0 | 0.00% |
| orders | order_amount | 53 | 0.41% |
| orders | delivery_minutes | 2,541 | 19.42% |

The 53 pandas-null order amounts are the `N/A` spellings; a further 67 `error` strings are non-numeric, giving 120 unparseable amounts in total. The 2,541 missing delivery times include cancelled orders and completed orders with missing delivery times.

| Raw audit checkpoint | Result |
|---|---:|
| orders rows | 13,084 |
| exact duplicate rows | 119 |
| pending rows | 656 |
| distinct order_id | 12,309 |
| non-numeric order_amount | 120 |
| negative delivery_minutes | 106 |
| blank delivery_minutes | 2,541 |
| blank customer ages | 61 |
| non-blank ages outside 10–100 | 6 |
| raw city spellings | 24 |
| distinct non-blank gender spellings | 10 |
| blank gender | 30 |

### Cleaning rules and changes

1. **Customers:** trimmed city whitespace and applied title case, consolidating 24 raw spellings into six cities. Mapped male/M/m/MALE variants to `male`, female/F/f/FEMALE variants to `female`, and blanks to `unknown`. Converted ages to numeric, changed the six impossible ages outside 10–100 to missing, then filled all 67 missing/impossible ages using each city's median with `groupby('city').transform('median')`. Converted `signup_date` to datetime. Final age nulls: zero; range: 18–60.
2. **Orders:** parsed `order_ts` as datetime; removed 119 exact duplicate rows; sorted by timestamp and kept the latest record per `order_id`, resolving 656 earlier pending versions and producing one final row per order. Converted `order_amount` to numeric (120 unparseable values); imputed all 120 from `round(price_inr × quantity × (1 − discount_pct/100), 2)`. Changed 106 negative delivery durations to missing. For completed orders, filled missing delivery durations using the completed-order median for the product category; cancelled orders were deliberately left without a delivery time. No pending rows remain.
3. **Products:** retained the supplied 40-row product catalogue as the clean product dimension.

Clean customers: 1,200 rows; city counts Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184. Gender counts: female 610, male 560, unknown 30. Clean orders: 12,309 rows; completed 11,110 and cancelled 1,199; completed delivery nulls 0, cancelled delivery nulls 1,199; order amount nulls 0. Completed-order revenue is **INR 2,998,668.80**.

`data/clean/{customers,products,orders}.csv` and `data/instamart.db` are generated by `src/part_a.py`. SQLite contains `customers`, `products` and `orders`; `order_ts` is stored as `YYYY-MM-DD HH:MM:SS` text. `validate.py` asserts more than eight reference checkpoints.

### SQL analysis

All seven SQL statements are saved in `sql/queries.sql` and were executed against the SQLite database using `pandas.read_sql_query`.

| Query | Question / method | Result |
|---|---|---:|
| Q1 | Completed orders above INR 500, grouped by customer, HAVING spend above INR 8,000 | 30 customers |
| Q2 | Revenue by city and product category, three-table join | 30 cells; total INR 2,998,668.80 |
| Q2 maximum | Bengaluru / Electronics | INR 233,909.45 |
| Q3 | Products above average catalogue price, scalar subquery | 11 products |
| Q4 | Customers with no completed order, LEFT JOIN | 25 customers |
| Q5 | Customers above average customer completed spend, two CTEs | 550 customers; mean spend INR 2,552.06 |
| Q6 | Per-city RANK, DENSE_RANK and ROW_NUMBER | 8 rows have RANK different from ROW_NUMBER |
| Q7 | Daily completed revenue and cumulative window sum | 183 days; final cumulative INR 2,998,668.80 |

Top completed-spend customer in each city:

| City | Customer | Spend |
|---|---|---:|
| Bengaluru | C0082 | INR 12,856.20 |
| Chennai | C0930 | INR 10,151.40 |
| Delhi | C0291 | INR 13,742.60 |
| Hyderabad | C0743 | INR 12,730.05 |
| Mumbai | C0693 | INR 14,766.55 |
| Pune | C0426 | INR 10,377.10 |

Ties receive the same rank in `RANK()` while `ROW_NUMBER()` assigns a distinct sequential number, so rows after a tie can have different rank and row-number values.

### pandas cross-check and exploration

A `pivot_table(..., margins=True)` reproduces the SQL city/category grand total of INR 2,998,668.80. Completed spend by city:

| City | Sum | Mean | Order count |
|---|---:|---:|---:|
| Bengaluru | INR 523,643.40 | INR 276.47 | 1,894 |
| Chennai | INR 463,629.15 | INR 272.72 | 1,700 |
| Delhi | INR 502,829.80 | INR 243.62 | 2,064 |
| Hyderabad | INR 528,589.90 | INR 284.95 | 1,855 |
| Mumbai | INR 479,058.50 | INR 297.00 | 1,613 |
| Pune | INR 500,918.05 | INR 252.48 | 1,984 |

Completed `order_amount` distribution: mean INR 269.91; median INR 135.00; skewness 4.61; Q1 INR 64.80; Q3 INR 297.00; IQR INR 232.20; lower fence INR −283.50; upper fence INR 645.30; outliers below lower fence 0; outliers above upper fence 897.

The dataset describes completed grocery and convenience orders over the six-month synthetic order window. Order amounts are strongly right-skewed: the mean (INR 269.91) is almost twice the median (INR 135.00), and 897 orders exceed the upper 1.5×IQR fence. These large values are plausible high-value baskets, particularly electronics, rather than automatically being data errors; the planted impossible delivery values and malformed amounts were handled during cleaning. The largest absolute correlation is between order amount and discount percentage (about −0.095), followed by amount and quantity (+0.087). A correlation of this size is weak and does not establish causation.

## Part B

### RFM and segmentation

RFM uses completed orders only, with snapshot `2025-10-01 00:00:00`. It contains 1,175 customers. Recency: median 19 days, minimum 0, maximum 180. Frequency: mean 9.46, maximum 33. Monetary: mean INR 2,552.06, maximum INR 14,766.55.

The three RFM features were standardised with `StandardScaler`; K-means used `n_init=10` and `random_state=42` for every K. Silhouette is computed on the scaled RFM matrix.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1,820.8 | 0.465 |
| 3 | 1,268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.481 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

K=4 was selected because it has the strongest silhouette in this run, a clear inertia bend, and yields four interpretable campaign groups. Inertia is the within-cluster sum of squared distances, so adding clusters can only keep it the same or reduce it; K=8 therefore has lower inertia than K=2 by construction. That monotonic decrease alone cannot identify a useful K because an excessively large K can simply split customers into many small groups without creating actionable business segments.

| Segment ID | Size | Mean recency (days) | Mean frequency | Mean monetary | Segment name | Recommended action |
|---:|---:|---:|---:|---:|---|---|
| 0 | 436 | 28.86 | 3.77 | INR 1,057.62 | Occasional Shoppers | Use category-led reminders and low-cost reorder nudges. |
| 1 | 138 | 24.12 | 6.77 | INR 7,129.70 | Premium Champions (**Premium**) | Offer concierge support, relevant electronics/personal-care bundles and early access. |
| 2 | 419 | 8.77 | 19.40 | INR 3,410.02 | Frequent Essentials | Protect habit with replenishment reminders and subscription-style offers. |
| 3 | 182 | 109.04 | 2.20 | INR 685.99 | At-Risk Customers | Trigger a win-back campaign with a modest personalised incentive. |

The `is_premium` flag is true only for the segment with the highest mean monetary value (segment 1). Cluster numbers are arbitrary labels, not an ordinal ranking. PCA on scaled RFM explains 0.6343 and 0.2180 of variance in its first two components, respectively (total 0.8523).

### Lapse cohort, features and leakage controls

The cutoff is `2025-09-01`. The cohort contains 1,156 customers with at least one completed pre-cutoff order; 443 are labelled lapsed (38.32%), with no completed order from September 1 through September 30. The stratified 80/20 split contains 924 training and 232 test rows (`random_state=42`).

Features use only orders dated before September 1 plus the clean customer table: recency, frequency, monetary, average discount, cancellation rate, average delivery time, favourite product category, city, age, gender and tenure. No order dated September 1 or later enters the feature matrix, and the RFM segment labels are not features. Numeric features are imputed/scaled and nominal features are one-hot encoded with `drop='first'` inside a `ColumnTransformer`/Pipeline. The pipelines and their scalers/encoders are fitted only on training data; GridSearchCV folds are also restricted to training data. The oversampler sees training rows only; no encoder, scaler or resampler sees test rows. Test rows are only transformed and used for final evaluation. A code assertion confirms there are no missing feature values. The training-fitted scaler's frequency mean is 8.115 versus 8.029 across the full cohort because the scaler sees only the training partition, not the held-out test customers.

Always predicting not-lapsed gives test accuracy 61.64% and lapsed-class F1 0.000 (the positive class is lapsed; metric calls use `zero_division=0`).

### Model tuning and test evaluation

Logistic regression used `max_iter=1000`, grid C ∈ {0.01, 0.1, 1, 10, 100}, five-fold CV and F1 scoring. Random forest used 100 trees, `random_state=42`, grid max_depth ∈ {3, 5, 8} × min_samples_leaf ∈ {5, 10, 20}, five-fold CV (45 fits).

| Model | Best parameters | Best CV F1 | Test confusion matrix (TN, FP / FN, TP) | Lapsed precision | Lapsed recall | Lapsed F1 |
|---|---|---:|---|---:|---:|---:|
| Logistic regression | C=10 | 0.6565 | [[104, 39], [32, 57]] | 0.5938 | 0.6404 | 0.6162 |
| Random forest | depth=8, leaf=10 | 0.6827 | [[113, 30], [31, 58]] | 0.6591 | 0.6517 | 0.6554 |

The random forest is selected because it has the stronger untouched-test F1 (0.655) and precision (0.659), while retaining similar recall. Missing a genuinely lapsing customer loses a retention opportunity and future value; wrongly targeting a loyal customer wastes incentive budget and can annoy them. In this experiment the forest offers a better balance of these errors than logistic regression. The model was saved to `outputs/lapse_model.joblib`, reloaded, and its test predictions matched the original model exactly.

### Class balance and interpretation

The training-only oversampling retained every original row and resampled lapsed training rows with replacement (`random_state=42`) until each class had 570 rows. On the unchanged test set, logistic regression moved from precision 0.594, recall 0.640, F1 0.616 to precision 0.595, recall 0.876, F1 0.709. Recall moved most: the retention team catches substantially more at-risk customers, at the cost of more false positives (53 instead of 39 for the logistic baseline). Precision stayed almost unchanged rather than improving.

Top random-forest importances: frequency 0.2827, monetary 0.2740, recency 0.1103, average delivery 0.0637, cancellation rate 0.0516. Frequency and monetary dominate the forest's decisions, consistent with the value and regularity of prior ordering. In the fitted logistic regression, frequency has a negative coefficient (−1.5889), recency positive (+0.0342), and monetary positive (+0.1994); holding other encoded/scaled features fixed, these signs describe model associations, not causal effects.

`outputs/retention_list.csv` contains the top 10 test customers who are Premium according to the whole-window RFM flag, ranked by predicted lapse probability; 1 of the 10 is actually lapsed. This retention list is illustrative and not a leakage-free evaluation because Premium is a post-hoc whole-window label that includes September activity; the model features themselves remain pre-cutoff only.

## Part C

### Guardrails, redaction and prompts

Tickets are processed in raw file order. Four empty/whitespace-only tickets are marked `skipped_empty`; four tickets longer than 400 estimated input tokens are marked `flagged_long` (734 tokens each). Neither class is sent to the model. Redaction is performed in the required order using the supplied regular expressions: email → `[EMAIL]`, phone → `[PHONE]`, order ID → `[ORDER_ID]`. Substitutions across all 164 tickets: 55 emails, 91 phone numbers and 132 order IDs (278 total). Reapplying all three patterns to every redacted text finds zero matches. The user message is exactly the redacted ticket text, with no extra wrapper text.

Both system prompts include `CLASSIFY`, the four allowed labels and a JSON-only response schema. The zero-shot prompt has no `EXAMPLES:` section. The few-shot prompt adds three labelled examples, including fraud, after an `EXAMPLES:` line. Every call has up to three retries after transient errors; exponential backoff uses `0.05 × 2**n × uniform(0.9, 1.15)` seconds. `mock_llm.reset()` runs before each batch. Each full run encountered and recovered 19 transient errors; each retry log line includes ticket ID, retry number and wait. The observed retry-0 waits fell between 0.04531 and 0.05615 seconds.

### Classification results

| Run | Classified | JSON | Keyword fallback | Invalid output | Accuracy among classified | Total tokens | Tokens / classified |
|---|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot | 147 | 135 | 12 | 9 | 85.71% (126/147) | 9,088 | 61.82 |
| Few-shot | 147 | 135 | 12 | 9 | 100.00% (147/147) | 14,548 | 98.97 |

Each run also has 4 `skipped_empty` and 4 `flagged_long` rows. Valid JSON categories are accepted as `classified_json`; a non-JSON response is accepted only when regex word extraction finds exactly one allowed category (`classified_keyword`). Invalid JSON categories and all other outputs go to human review as `flagged_invalid_output`.

Zero-shot confusion table (rows are true category; columns are predicted category; only the 147 classified tickets are counted):

| True \ Predicted | billing | delivery | technical | fraud |
|---|---:|---:|---:|---:|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| technical | 10 | 0 | 29 | 0 |
| fraud | 11 | 0 | 0 | 23 |

The few-shot prompt costs 5,460 additional tokens overall, or 37.14 more tokens per classified ticket, but increases measured accuracy by 14.29 percentage points (85.71% to 100%). It is worth paying that cost in this safety-sensitive routing workflow because the zero-shot errors are concentrated in fraud (11) and technical (10) tickets mislabelled as billing; correct few-shot routing avoids sending these sensitive cases down the wrong queue. Outputs that passed the explicit parser are rule-validated; empty, long, invalid-output and budget-exhausted cases are sent to human review.

### Budget and routing

The zero-shot budget demonstration used TOKEN_BUDGET=6,000 and marked 53 tickets `budget_exhausted`. Before each eligible model call, the pipeline checks whether accumulated usage is already at or above the budget. The final few-shot run uses TOKEN_BUDGET=100,000 and writes all 164 rows to `outputs/triage_results.csv`. Routing precedence is: guardrail/invalid/budget statuses → `human_review`/NORMAL; fraud → `security_team`/HIGH; Premium customer → `premium_desk`/HIGH; otherwise `support_queue`/NORMAL. Missing segment matches are treated as not Premium. `outputs/triage_budget_demo.csv` contains the budget demonstration.

## Part D

### Policy retrieval and grounded answers

The five policy files are split into 40-word chunks with a 10-word overlap (step size 30), preserving the source filename. The result is 15 chunks: refund 4, delivery 3, escalation 3, payment 3, privacy 2. `TfidfVectorizer()` is fitted on chunk text; query vectors use that same fit, and cosine similarity ranks chunks. For `refund policy late delivery`, all top three retrieved chunks come from `refund_policy.txt`. TF measures how often a term appears in a chunk; IDF downweights terms common across many chunks. TF-IDF is lexical, so semantically equivalent phrases that use different words may not retrieve each other well.

`answer(question)` retrieves three chunks and supplies only those source-labelled chunks as context to the offline `POLICY ASSISTANT`; the assistant must answer from context or return its fallback. Evaluation on 14 questions:

| Chunking | Answerable with expected document in top 3 | Expected fact substring present | Unanswerable questions with fallback |
|---|---:|---:|---:|
| 40 words / 10 overlap | 12/12 | 12/12 | 2/2 |
| 40 words / 0 overlap | 12/12 | 10/12 | 2/2 |

With zero overlap, the questions asking which delivery time qualifies for a refund and how soon a failed-but-deducted payment is returned lose the expected fact in the generated answer. The relevant sentence is split at a chunk boundary, demonstrating why overlap helps retain complete policy statements.

### Refund tools and ReAct execution

The tool map contains five tools: `get_order` (SQLite order lookup), `search_policy` (the same top-three retrieval), `calculate_refund` (100% of amount only when completed and delivered after 45 minutes), `issue_refund` (append an issue action to the JSONL ledger) and `escalate_to_human` (append a human-approval action). Only completed orders qualify; delivery in 45 minutes or less gives INR 0; amounts above INR 1,000 are escalated, never automatically issued. The tool specifications and callable map are in `src/part_d.py`.

| Scenario | Status | Refund / escalation amount | Model calls | Tokens |
|---|---|---:|---:|---:|
| S1 | REFUND | INR 128.25 | 5 | 947 |
| S2 | REFUND | INR 135.00 | 5 | 947 |
| S3 | ESCALATE | INR 1,999.00 | 5 | 934 |
| S4 | DENIED | INR 0.00 | 4 | 645 |
| S5 | DENIED | INR 0.00 | 3 | 367 |
| S6 | NOT_FOUND | INR 0.00 | 2 | 92 |
| S7 | FALLBACK_MAX_ITERATIONS | INR 0.00 | 6 | 3,060 |

The default maximum is six model calls, token limit 6,000 and wall-clock limit 10 seconds. S7 requests a tool repeatedly and safely ends as `FALLBACK_MAX_ITERATIONS` after six calls. The S1 token-limit demonstration (`token_limit=200`) ended `FALLBACK_TOKEN_LIMIT`; the zero-second time-limit demonstration ended `FALLBACK_TIMEOUT` before its first model call. Fallbacks return completed steps and a human-handoff summary. The normal scenario outputs are in `outputs/agent_runs.jsonl`; issued and escalated actions are recorded in `outputs/refund_ledger.jsonl`.

### Crash and resume

After each tool observation is appended, the agent writes a JSON checkpoint under `state/` with run/scenario IDs, complete message history, token usage, step count and executed-tool list. The S1 crash was injected immediately after the second tool execution and checkpoint write. Resuming that checkpoint completed with `REFUND: INR 128.25`. The tools executed across the crash and resume, in order, were exactly: `get_order`, `search_policy`, `calculate_refund`, `issue_refund` — four executions, with no repeated tool execution.

### ReAct versus plan-and-execute

ReAct used 5 calls for S1, 3 for S5 and 2 for S6, because each observation informed the next tool request. This costs more model round trips and tokens than a plan-and-execute design that asks for a full tool sequence once, but it reacts naturally to an unexpected `NOT_FOUND` result or a cancelled order: it can stop rather than continue a stale plan. A one-shot plan can be cheaper for the common happy path, but it must add validation and replanning logic for uncertain observations. For refund decisions involving money and escalation thresholds, I would choose ReAct with strict tool-side checks and bounded iterations because each step is verified against the database and policy before the next action.

## Limitations and Next Steps

The data, tickets, policies and model are synthetic; performance does not establish production readiness. TF-IDF has lexical retrieval limitations, the mock LLM is deterministic and cannot represent real language-model quality, and the lapse label covers a short observation window. The model's Premium retention list is illustrative and not a leakage-free evaluation because the Premium segment flag is a post-hoc whole-window label incorporating September orders, although all model features are pre-cutoff. Before deployment, validate on real governed data, review fairness and calibration, monitor drift, add human approval and audit controls, and evaluate retrieval/model quality with domain experts.
