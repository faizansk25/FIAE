# Transformation and Feature Algorithm Catalog

> **Purpose:** Define precise feature-operation semantics, safeguards, leakage behavior, cost, and validation rules, including explicit shift/lag/rolling steps.

**Status:** Normative specification for one continuously evolving FIAE codebase. These documents describe implementation gates, not user-facing versions.


## Catalog contract

Every operation declares:

1. input semantic types;
2. output semantic type;
3. purpose;
4. preconditions;
5. learned fit state;
6. exact transform;
7. null/invalid behavior;
8. leakage class;
9. computational shape;
10. generation trigger;
11. rejection conditions;
12. validation requirement;
13. inference-state requirement;
14. tests.

Leakage classes:
- `L0`: deterministic row-wise/no learned population state;
- `L1`: learned unsupervised state; fit in training fold;
- `L2`: target-aware; cross-fit;
- `L3`: temporal/history; availability-aware;
- `L4`: semantic risk requiring review;
- `L5`: confirmed/future leakage; forbidden.

## 1. `identity`

- **Family:** numeric
- **Input:** numeric
- **Output:** numeric
- **Purpose:** retain raw baseline signal
- **Preconditions:** finite/parseable numeric or configured missing
- **Fit state:** none
- **Transform:** return x
- **Null / invalid policy:** preserve missing
- **Leakage class:** L0
- **Cost shape:** O(n), view if possible
- **Generate when:** all allowed numeric raw features
- **Reject when:** constant, leakage-rejected, identifier semantics
- **Validation:** baseline reference
- **Inference state:** input available
- **Mandatory tests:** roundtrip; null preservation

## 2. `log1p`

- **Family:** numeric
- **Input:** numeric
- **Output:** numeric
- **Purpose:** compress right skew and multiplicative scale
- **Preconditions:** x >= -1; normally non-negative domain
- **Fit state:** none
- **Transform:** z = log(1+x)
- **Null / invalid policy:** x<-1 invalid; null preserved
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** non-negative skew/heavy tail
- **Reject when:** domain violation or no useful variation
- **Validation:** incremental CV
- **Inference state:** input available
- **Mandatory tests:** domain; monotonicity; finite outputs

## 3. `signed_log1p`

- **Family:** numeric
- **Input:** numeric
- **Output:** numeric
- **Purpose:** compress positive and negative heavy tails
- **Preconditions:** finite numeric
- **Fit state:** none
- **Transform:** sign(x)*log(1+abs(x))
- **Null / invalid policy:** null preserved
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** mixed-sign heavy tail
- **Reject when:** no gain/constant
- **Validation:** incremental CV
- **Inference state:** input available
- **Mandatory tests:** sign preservation; inverse-order around zero

## 4. `sqrt`

- **Family:** numeric
- **Input:** numeric
- **Output:** numeric
- **Purpose:** stabilize count-like/non-negative skew
- **Preconditions:** x>=0
- **Fit state:** none
- **Transform:** sqrt(x)
- **Null / invalid policy:** negative invalid/null policy
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** counts/non-negative values
- **Reject when:** negative rate > tolerance
- **Validation:** incremental CV
- **Inference state:** input available
- **Mandatory tests:** domain; sqrt(0)

## 5. `cbrt`

- **Family:** numeric
- **Input:** numeric
- **Output:** numeric
- **Purpose:** mild signed tail compression
- **Preconditions:** finite numeric
- **Fit state:** none
- **Transform:** real cube root
- **Null / invalid policy:** null preserved
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** signed skew
- **Reject when:** no gain
- **Validation:** incremental CV
- **Inference state:** input available
- **Mandatory tests:** negative values supported

## 6. `square`

- **Family:** numeric
- **Input:** numeric
- **Output:** numeric
- **Purpose:** capture symmetric quadratic relationship
- **Preconditions:** finite and scale safe
- **Fit state:** none
- **Transform:** x*x with overflow check
- **Null / invalid policy:** overflow invalid
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** linear model residual curvature
- **Reject when:** overflow/extreme scale/no gain
- **Validation:** incremental CV
- **Inference state:** input available
- **Mandatory tests:** overflow guard

## 7. `cube`

- **Family:** numeric
- **Input:** numeric
- **Output:** numeric
- **Purpose:** capture odd nonlinear relationship
- **Preconditions:** finite and scale safe
- **Fit state:** none
- **Transform:** x*x*x with overflow check
- **Null / invalid policy:** overflow invalid
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** strong residual evidence
- **Reject when:** depth/cost/no gain
- **Validation:** incremental CV
- **Inference state:** input available
- **Mandatory tests:** overflow guard

## 8. `abs`

- **Family:** numeric
- **Input:** numeric
- **Output:** numeric
- **Purpose:** capture magnitude regardless of sign
- **Preconditions:** numeric
- **Fit state:** none
- **Transform:** abs(x)
- **Null / invalid policy:** null preserved
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** signed deviations
- **Reject when:** sign itself carries most signal/no gain
- **Validation:** incremental CV
- **Inference state:** input available
- **Mandatory tests:** idempotence

## 9. `sign`

- **Family:** numeric
- **Input:** numeric
- **Output:** categorical/ordinal
- **Purpose:** separate negative/zero/positive regimes
- **Preconditions:** numeric
- **Fit state:** none
- **Transform:** -1 if x<0, 0 if x=0, +1 if x>0
- **Null / invalid policy:** null separate
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** mixed-sign sparse features
- **Reject when:** single sign/no variance
- **Validation:** incremental CV
- **Inference state:** input available
- **Mandatory tests:** three-way mapping

## 10. `reciprocal`

- **Family:** numeric
- **Input:** numeric
- **Output:** numeric
- **Purpose:** model inverse rate/time relationships
- **Preconditions:** abs(x)>epsilon or safe-zero policy
- **Fit state:** none
- **Transform:** 1/x
- **Null / invalid policy:** zero/near-zero -> missing or dedicated indicator
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** domain suggests inverse
- **Reject when:** too many near-zero values
- **Validation:** incremental CV
- **Inference state:** input available
- **Mandatory tests:** zero policy

## 11. `exp_clip`

- **Family:** numeric
- **Input:** numeric
- **Output:** numeric
- **Purpose:** capture exponential response under bounded input
- **Preconditions:** bounded x range
- **Fit state:** none
- **Transform:** exp(clip(x,lo,hi))
- **Null / invalid policy:** overflow prevented by clip
- **Leakage class:** L0/L1 if learned clip
- **Cost shape:** O(n)
- **Generate when:** strong domain/prior evidence only
- **Reject when:** wide unbounded range/no evidence
- **Validation:** incremental CV
- **Inference state:** store clip if learned
- **Mandatory tests:** finite output

## 12. `standardize`

- **Family:** numeric
- **Input:** numeric
- **Output:** numeric
- **Purpose:** zero mean/unit variance for scale-sensitive models
- **Preconditions:** non-constant
- **Fit state:** learn mean/std on training partition
- **Transform:** (x-mean)/std
- **Null / invalid policy:** impute before or preserve missing based pipeline; std=0 reject
- **Leakage class:** L1
- **Cost shape:** O(n)
- **Generate when:** linear/SVM/KNN/PCA/neural
- **Reject when:** tree-only and no need
- **Validation:** inside-fold
- **Inference state:** store mean/std
- **Mandatory tests:** train-only fit; serialization

## 13. `robust_scale`

- **Family:** numeric
- **Input:** numeric
- **Output:** numeric
- **Purpose:** scale by median/IQR
- **Preconditions:** non-constant
- **Fit state:** learn median/IQR on training
- **Transform:** (x-median)/IQR
- **Null / invalid policy:** IQR=0 fallback/reject
- **Leakage class:** L1
- **Cost shape:** O(n)+quantile
- **Generate when:** outlier-heavy scale-sensitive path
- **Reject when:** degenerate/tiny sample
- **Validation:** inside-fold
- **Inference state:** store median/IQR
- **Mandatory tests:** fold isolation

## 14. `minmax_scale`

- **Family:** numeric
- **Input:** numeric
- **Output:** numeric
- **Purpose:** map training range to fixed interval
- **Preconditions:** non-constant
- **Fit state:** learn min/max
- **Transform:** (x-min)/(max-min)
- **Null / invalid policy:** out-of-range inference preserved or clipped by policy
- **Leakage class:** L1
- **Cost shape:** O(n)
- **Generate when:** bounded-input algorithms
- **Reject when:** extreme outlier domination/no need
- **Validation:** inside-fold
- **Inference state:** store min/max
- **Mandatory tests:** unknown-range behavior

## 15. `winsorize`

- **Family:** numeric
- **Input:** numeric
- **Output:** numeric
- **Purpose:** limit extreme tails
- **Preconditions:** enough data for stable quantiles
- **Fit state:** learn lower/upper quantiles
- **Transform:** clip to q_low/q_high
- **Null / invalid policy:** null preserved
- **Leakage class:** L1
- **Cost shape:** O(n)+quantile
- **Generate when:** heavy tails/outlier sensitivity
- **Reject when:** extremes meaningful/no gain/tiny data
- **Validation:** inside-fold
- **Inference state:** store thresholds
- **Mandatory tests:** fold-only quantiles

## 16. `quantile_normal`

- **Family:** numeric
- **Input:** numeric
- **Output:** numeric
- **Purpose:** map empirical distribution toward normal
- **Preconditions:** enough unique values
- **Fit state:** learn empirical quantile map
- **Transform:** rank/CDF -> normal quantile
- **Null / invalid policy:** outside range -> boundary/interpolation
- **Leakage class:** L1
- **Cost shape:** O(n log n) fit
- **Generate when:** non-Gaussian + compatible model
- **Reject when:** small n/high cost
- **Validation:** inside-fold
- **Inference state:** store quantile map
- **Mandatory tests:** monotonic mapping

## 17. `zero_indicator`

- **Family:** numeric
- **Input:** numeric
- **Output:** boolean
- **Purpose:** capture structural zeros
- **Preconditions:** zero fraction > 0
- **Fit state:** none
- **Transform:** x==0
- **Null / invalid policy:** null separate
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** excess zeros
- **Reject when:** zero rare/no variance
- **Validation:** incremental CV
- **Inference state:** input available
- **Mandatory tests:** zero/missing distinction

## 18. `positive_indicator`

- **Family:** numeric
- **Input:** numeric
- **Output:** boolean
- **Purpose:** capture sign regime
- **Preconditions:** both positive and non-positive values
- **Fit state:** none
- **Transform:** x>0
- **Null / invalid policy:** null separate
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** threshold/regime behavior
- **Reject when:** no variation
- **Validation:** incremental CV
- **Inference state:** input available
- **Mandatory tests:** boundary

## 19. `missing_indicator`

- **Family:** missingness
- **Input:** any
- **Output:** boolean
- **Purpose:** capture informative missingness
- **Preconditions:** missing exists
- **Fit state:** none
- **Transform:** is_missing(x)
- **Null / invalid policy:** always defined
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** any missing feature
- **Reject when:** no missing values
- **Validation:** incremental CV
- **Inference state:** same missing semantics
- **Mandatory tests:** missing-token consistency

## 20. `finite_indicator`

- **Family:** numeric
- **Input:** numeric
- **Output:** boolean
- **Purpose:** detect invalid numeric values
- **Preconditions:** NaN/inf possible
- **Fit state:** none
- **Transform:** isfinite(x)
- **Null / invalid policy:** always defined
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** messy numerical sources
- **Reject when:** all finite/no variance
- **Validation:** incremental CV
- **Inference state:** same parse semantics
- **Mandatory tests:** NaN/inf handling

## 21. `sum`

- **Family:** numeric interaction
- **Input:** numeric,numeric
- **Output:** numeric
- **Purpose:** add compatible quantities
- **Preconditions:** compatible units/evidence
- **Fit state:** none
- **Transform:** a+b
- **Null / invalid policy:** propagate missing unless configured
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** semantic pair/prior
- **Reject when:** unit mismatch/arbitrary combinatorial pair
- **Validation:** incremental CV
- **Inference state:** both available
- **Mandatory tests:** commutative canonicalization

## 22. `difference`

- **Family:** numeric interaction
- **Input:** numeric,numeric
- **Output:** numeric
- **Purpose:** measure gap/net/change
- **Preconditions:** ordered compatible units
- **Fit state:** none
- **Transform:** a-b
- **Null / invalid policy:** propagate missing
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** actual-plan, price-cost, end-start
- **Reject when:** no meaningful direction
- **Validation:** incremental CV
- **Inference state:** both available
- **Mandatory tests:** non-commutative signature

## 23. `product`

- **Family:** numeric interaction
- **Input:** numeric,numeric
- **Output:** numeric
- **Purpose:** capture multiplicative interaction
- **Preconditions:** finite scale
- **Fit state:** none
- **Transform:** a*b with overflow guard
- **Null / invalid policy:** overflow invalid
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** residual/domain interaction
- **Reject when:** overflow/combinatorial explosion
- **Validation:** incremental CV
- **Inference state:** both available
- **Mandatory tests:** overflow

## 24. `safe_ratio`

- **Family:** numeric interaction
- **Input:** numeric,numeric
- **Output:** numeric
- **Purpose:** normalize numerator by denominator
- **Preconditions:** denominator meaningful
- **Fit state:** none
- **Transform:** a/b when abs(b)>eps else missing/special
- **Null / invalid policy:** track zero denominator
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** amount/count, distance/time
- **Reject when:** too many zero denominators
- **Validation:** incremental CV + invalid penalty
- **Inference state:** both available
- **Mandatory tests:** division zero; sign

## 25. `relative_difference`

- **Family:** numeric interaction
- **Input:** numeric,numeric
- **Output:** numeric
- **Purpose:** scale difference by reference magnitude
- **Preconditions:** reference denominator meaningful
- **Fit state:** none
- **Transform:** (a-b)/(abs(b)+eps)
- **Null / invalid policy:** missing propagate
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** relative performance
- **Reject when:** near-zero reference dominates
- **Validation:** incremental CV
- **Inference state:** both available
- **Mandatory tests:** zero policy

## 26. `min_pair`

- **Family:** numeric interaction
- **Input:** numeric,numeric
- **Output:** numeric
- **Purpose:** lower envelope/bottleneck
- **Preconditions:** numeric
- **Fit state:** none
- **Transform:** min(a,b)
- **Null / invalid policy:** explicit missing policy
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** capacity/bottleneck prior
- **Reject when:** high redundancy/no evidence
- **Validation:** incremental CV
- **Inference state:** both available
- **Mandatory tests:** commutative

## 27. `max_pair`

- **Family:** numeric interaction
- **Input:** numeric,numeric
- **Output:** numeric
- **Purpose:** upper envelope/capacity
- **Preconditions:** numeric
- **Fit state:** none
- **Transform:** max(a,b)
- **Null / invalid policy:** explicit missing policy
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** maximum regime
- **Reject when:** redundant/no evidence
- **Validation:** incremental CV
- **Inference state:** both available
- **Mandatory tests:** commutative

## 28. `mean_pair`

- **Family:** numeric interaction
- **Input:** numeric,numeric
- **Output:** numeric
- **Purpose:** combine repeated measurements
- **Preconditions:** compatible scale
- **Fit state:** none
- **Transform:** (a+b)/2
- **Null / invalid policy:** missing policy
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** redundant measurements
- **Reject when:** unit mismatch
- **Validation:** incremental CV
- **Inference state:** both available
- **Mandatory tests:** symmetry

## 29. `harmonic_mean_pair`

- **Family:** numeric interaction
- **Input:** positive numeric,positive numeric
- **Output:** numeric
- **Purpose:** emphasize lower of two rates
- **Preconditions:** strictly positive
- **Fit state:** none
- **Transform:** 2ab/(a+b)
- **Null / invalid policy:** zero/negative invalid
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** rate/bottleneck semantics
- **Reject when:** domain violation
- **Validation:** incremental CV
- **Inference state:** both available
- **Mandatory tests:** zero denominator

## 30. `geometric_mean_pair`

- **Family:** numeric interaction
- **Input:** non-negative numeric pair
- **Output:** numeric
- **Purpose:** multiplicative central tendency
- **Preconditions:** a,b>=0
- **Fit state:** none
- **Transform:** sqrt(a*b) stable
- **Null / invalid policy:** negative invalid
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** scale ratios/multiplicative domains
- **Reject when:** domain/no evidence
- **Validation:** incremental CV
- **Inference state:** both available
- **Mandatory tests:** overflow-safe

## 31. `euclidean_norm_pair`

- **Family:** numeric interaction
- **Input:** numeric,numeric
- **Output:** numeric
- **Purpose:** 2D magnitude
- **Preconditions:** comparable/scaled dimensions
- **Fit state:** none
- **Transform:** hypot(a,b)
- **Null / invalid policy:** missing propagate
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** coordinates/deviation
- **Reject when:** scale mismatch
- **Validation:** incremental CV
- **Inference state:** both available
- **Mandatory tests:** stable hypot

## 32. `absolute_difference`

- **Family:** numeric interaction
- **Input:** numeric,numeric
- **Output:** numeric
- **Purpose:** distance between values
- **Preconditions:** comparable units
- **Fit state:** none
- **Transform:** abs(a-b)
- **Null / invalid policy:** missing propagate
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** agreement/deviation
- **Reject when:** direction important/no evidence
- **Validation:** incremental CV
- **Inference state:** both available
- **Mandatory tests:** symmetry

## 33. `one_hot`

- **Family:** categorical
- **Input:** categorical
- **Output:** sparse boolean
- **Purpose:** independent category indicators
- **Preconditions:** cardinality under dimensional budget
- **Fit state:** learn training vocabulary
- **Transform:** one binary column per retained category
- **Null / invalid policy:** unknown -> all-zero/unknown bucket; missing explicit
- **Leakage class:** L1
- **Cost shape:** O(n*k) logical; sparse preferred
- **Generate when:** low/moderate cardinality
- **Reject when:** dimension/RAM too high
- **Validation:** inside-fold vocabulary
- **Inference state:** store categories/order
- **Mandatory tests:** unknown category

## 34. `ordinal_encode`

- **Family:** categorical
- **Input:** categorical
- **Output:** integer
- **Purpose:** compact representation
- **Preconditions:** categorical
- **Fit state:** learn mapping unless true order provided
- **Transform:** map category->code
- **Null / invalid policy:** unknown reserved code
- **Leakage class:** L1
- **Cost shape:** O(n)
- **Generate when:** tree learner/true ordinal
- **Reject when:** linear model fake order risk
- **Validation:** inside-fold
- **Inference state:** store mapping
- **Mandatory tests:** unknown code

## 35. `frequency_encode`

- **Family:** categorical
- **Input:** categorical
- **Output:** numeric
- **Purpose:** encode prevalence
- **Preconditions:** categorical
- **Fit state:** learn count/N on training
- **Transform:** map category->frequency
- **Null / invalid policy:** unknown -> 0/prior
- **Leakage class:** L1
- **Cost shape:** O(n)
- **Generate when:** high cardinality
- **Reject when:** unstable shift/no gain
- **Validation:** inside-fold
- **Inference state:** store map
- **Mandatory tests:** fold isolation

## 36. `count_encode`

- **Family:** categorical
- **Input:** categorical
- **Output:** numeric
- **Purpose:** encode support count
- **Preconditions:** categorical
- **Fit state:** learn counts on training
- **Transform:** map category->count
- **Null / invalid policy:** unknown -> 0
- **Leakage class:** L1
- **Cost shape:** O(n)
- **Generate when:** high cardinality
- **Reject when:** temporal future-count leakage
- **Validation:** fold/time safe
- **Inference state:** store map
- **Mandatory tests:** history semantics

## 37. `rare_group`

- **Family:** categorical
- **Input:** categorical
- **Output:** categorical
- **Purpose:** collapse tail categories
- **Preconditions:** many low-count levels
- **Fit state:** learn retained category set
- **Transform:** rare -> __RARE__
- **Null / invalid policy:** missing separate
- **Leakage class:** L1
- **Cost shape:** O(n)
- **Generate when:** long tail
- **Reject when:** important rare signal harmed
- **Validation:** inside-fold
- **Inference state:** store retained set
- **Mandatory tests:** unknown handling

## 38. `hash_encode`

- **Family:** categorical
- **Input:** categorical
- **Output:** fixed sparse numeric
- **Purpose:** bounded-memory category representation
- **Preconditions:** hash acceptable
- **Fit state:** fixed seed/dimension only
- **Transform:** hash category to bucket; optional signed hashing
- **Null / invalid policy:** missing dedicated token
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** very high cardinality
- **Reject when:** collision/interpretability constraint
- **Validation:** incremental CV
- **Inference state:** same seed/dim
- **Mandatory tests:** deterministic hash

## 39. `category_cross`

- **Family:** categorical interaction
- **Input:** categorical,categorical
- **Output:** categorical/hash
- **Purpose:** capture joint category state
- **Preconditions:** joint cardinality estimated safe or hashed
- **Fit state:** learn vocab if one-hot/frequency path
- **Transform:** canonical escaped pair
- **Null / invalid policy:** missing tokens explicit
- **Leakage class:** L0/L1
- **Cost shape:** O(n), cardinality can explode
- **Generate when:** historical/interaction evidence
- **Reject when:** joint cardinality too high
- **Validation:** inside-fold when state learned
- **Inference state:** same constructor
- **Mandatory tests:** escaping; commutativity policy

## 40. `target_mean_crossfit`

- **Family:** target-aware categorical
- **Input:** categorical,target
- **Output:** numeric
- **Purpose:** encode expected target by category
- **Preconditions:** target-aware allowed; enough support
- **Fit state:** learn per training fold with smoothing
- **Transform:** OOF category mean; final dev map for inference
- **Null / invalid policy:** unknown -> global prior
- **Leakage class:** L2
- **Cost shape:** O(n)
- **Generate when:** high-cardinality with target signal
- **Reject when:** tiny groups/leakage/stability failure
- **Validation:** strict cross-fit
- **Inference state:** store final map/prior
- **Mandatory tests:** self-target exclusion

## 41. `woe_crossfit`

- **Family:** target-aware categorical
- **Input:** categorical,binary target
- **Output:** numeric
- **Purpose:** log class-distribution evidence
- **Preconditions:** binary target; enough positives/negatives
- **Fit state:** learn per fold with smoothing
- **Transform:** log((p_pos+eps)/(p_neg+eps)) by category
- **Null / invalid policy:** unknown -> global prior
- **Leakage class:** L2
- **Cost shape:** O(n)
- **Generate when:** binary risk-style tasks
- **Reject when:** zero support/tiny class counts
- **Validation:** strict stratified cross-fit
- **Inference state:** store maps
- **Mandatory tests:** smoothing; finite output

## 42. `numeric_to_cat_target`

- **Family:** target-aware mixed
- **Input:** numeric,target
- **Output:** numeric/categorical
- **Purpose:** discover nonlinear target regions
- **Preconditions:** enough data
- **Fit state:** learn bins + target means inside fold
- **Transform:** bin numeric then cross-fit encode bins
- **Null / invalid policy:** missing bin/prior
- **Leakage class:** L2
- **Cost shape:** model-dependent
- **Generate when:** nonlinear target relationship
- **Reject when:** small data/leakage
- **Validation:** strict cross-fit
- **Inference state:** store final bins/map
- **Mandatory tests:** train-only bins

## 43. `ordinal_true_scale`

- **Family:** categorical
- **Input:** ordered categorical
- **Output:** numeric
- **Purpose:** respect known ordered semantics
- **Preconditions:** explicit ordering supplied
- **Fit state:** use fixed user/domain order
- **Transform:** map levels to monotone rank
- **Null / invalid policy:** unknown invalid/prior
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** true ordinal feature
- **Reject when:** order inferred only from target
- **Validation:** incremental CV
- **Inference state:** same fixed order
- **Mandatory tests:** order validation

## 44. `year`

- **Family:** datetime
- **Input:** datetime
- **Output:** numeric/categorical
- **Purpose:** calendar year
- **Preconditions:** valid datetime
- **Fit state:** none
- **Transform:** extract year
- **Null / invalid policy:** missing preserved
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** multi-year data
- **Reject when:** constant
- **Validation:** incremental CV
- **Inference state:** timestamp available
- **Mandatory tests:** timezone consistency

## 45. `quarter`

- **Family:** datetime
- **Input:** datetime
- **Output:** categorical
- **Purpose:** quarter seasonality
- **Preconditions:** valid datetime
- **Fit state:** none
- **Transform:** 1..4
- **Null / invalid policy:** missing preserved
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** multi-quarter span
- **Reject when:** constant
- **Validation:** incremental CV
- **Inference state:** timestamp available
- **Mandatory tests:** boundary months

## 46. `month`

- **Family:** datetime
- **Input:** datetime
- **Output:** categorical
- **Purpose:** month seasonality
- **Preconditions:** valid datetime
- **Fit state:** none
- **Transform:** 1..12
- **Null / invalid policy:** missing preserved
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** multi-month span
- **Reject when:** constant
- **Validation:** incremental CV
- **Inference state:** timestamp available
- **Mandatory tests:** timezone

## 47. `week_of_year`

- **Family:** datetime
- **Input:** datetime
- **Output:** numeric/categorical
- **Purpose:** annual week pattern
- **Preconditions:** valid datetime
- **Fit state:** none
- **Transform:** ISO week
- **Null / invalid policy:** missing preserved
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** weekly seasonality
- **Reject when:** constant/no span
- **Validation:** incremental CV
- **Inference state:** timestamp available
- **Mandatory tests:** year-boundary ISO

## 48. `day_of_week`

- **Family:** datetime
- **Input:** datetime
- **Output:** categorical
- **Purpose:** weekly cycle
- **Preconditions:** valid datetime
- **Fit state:** none
- **Transform:** fixed convention 0..6
- **Null / invalid policy:** missing preserved
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** daily/subdaily
- **Reject when:** constant
- **Validation:** incremental CV
- **Inference state:** timestamp available
- **Mandatory tests:** locale convention

## 49. `day_of_month`

- **Family:** datetime
- **Input:** datetime
- **Output:** numeric
- **Purpose:** month position
- **Preconditions:** valid datetime
- **Fit state:** none
- **Transform:** 1..31
- **Null / invalid policy:** missing preserved
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** billing/pay cycles
- **Reject when:** no variation
- **Validation:** incremental CV
- **Inference state:** timestamp available
- **Mandatory tests:** month length

## 50. `day_of_year`

- **Family:** datetime
- **Input:** datetime
- **Output:** numeric
- **Purpose:** annual position
- **Preconditions:** valid datetime
- **Fit state:** none
- **Transform:** 1..365/366
- **Null / invalid policy:** missing preserved
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** annual pattern
- **Reject when:** short span
- **Validation:** incremental CV
- **Inference state:** timestamp available
- **Mandatory tests:** leap year

## 51. `hour`

- **Family:** datetime
- **Input:** datetime
- **Output:** categorical
- **Purpose:** intraday cycle
- **Preconditions:** subday time
- **Fit state:** none
- **Transform:** 0..23
- **Null / invalid policy:** missing preserved
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** intraday data
- **Reject when:** constant
- **Validation:** incremental CV
- **Inference state:** timestamp available
- **Mandatory tests:** timezone

## 52. `minute`

- **Family:** datetime
- **Input:** datetime
- **Output:** categorical/numeric
- **Purpose:** within-hour pattern
- **Preconditions:** minute resolution
- **Fit state:** none
- **Transform:** 0..59
- **Null / invalid policy:** missing preserved
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** fine-grained events
- **Reject when:** high noise/no variation
- **Validation:** incremental CV
- **Inference state:** timestamp available
- **Mandatory tests:** resolution

## 53. `is_weekend`

- **Family:** datetime
- **Input:** datetime
- **Output:** boolean
- **Purpose:** weekend regime
- **Preconditions:** calendar/locale policy known
- **Fit state:** none
- **Transform:** weekday in weekend set
- **Null / invalid policy:** missing preserved
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** business activity
- **Reject when:** locale ambiguous/no gain
- **Validation:** incremental CV
- **Inference state:** calendar policy
- **Mandatory tests:** weekend set

## 54. `cyclical_month`

- **Family:** datetime
- **Input:** datetime
- **Output:** 2 numeric
- **Purpose:** circular month distance
- **Preconditions:** month valid
- **Fit state:** none
- **Transform:** sin(2πm/12), cos(2πm/12)
- **Null / invalid policy:** pair missing
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** linear/distance models
- **Reject when:** tree-only no gain
- **Validation:** incremental CV
- **Inference state:** timestamp available
- **Mandatory tests:** norm approx 1

## 55. `cyclical_dow`

- **Family:** datetime
- **Input:** datetime
- **Output:** 2 numeric
- **Purpose:** circular weekday
- **Preconditions:** dow valid
- **Fit state:** none
- **Transform:** sin(2πd/7), cos(2πd/7)
- **Null / invalid policy:** pair missing
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** cyclic weekly
- **Reject when:** no gain
- **Validation:** incremental CV
- **Inference state:** timestamp available
- **Mandatory tests:** period 7

## 56. `cyclical_hour`

- **Family:** datetime
- **Input:** datetime
- **Output:** 2 numeric
- **Purpose:** circular hour
- **Preconditions:** hour valid
- **Fit state:** none
- **Transform:** sin(2πh/24), cos(2πh/24)
- **Null / invalid policy:** pair missing
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** cyclic intraday
- **Reject when:** no gain
- **Validation:** incremental CV
- **Inference state:** timestamp available
- **Mandatory tests:** period 24

## 57. `elapsed_since_reference`

- **Family:** datetime interaction
- **Input:** datetime,datetime
- **Output:** numeric
- **Purpose:** tenure/recency
- **Preconditions:** reference available by prediction time
- **Fit state:** none/fixed semantics
- **Transform:** (event-reference) in chosen units
- **Null / invalid policy:** missing propagate
- **Leakage class:** L0
- **Cost shape:** O(n)
- **Generate when:** signup age/elapsed time
- **Reject when:** reference is future/post-outcome
- **Validation:** incremental CV
- **Inference state:** both times available
- **Mandatory tests:** unit/timezone

## 58. `shift_lag_rows`

- **Family:** temporal
- **Input:** value,time[,group]
- **Output:** same as value
- **Purpose:** previous ordered observation
- **Preconditions:** strict order; lag k>=1
- **Fit state:** none for source values; history state at inference
- **Transform:** stable-sort by group,time,tie; output value i-k
- **Null / invalid policy:** first k per group missing
- **Leakage class:** L3
- **Cost shape:** O(n log n) sort or O(n) presorted
- **Generate when:** autocorrelation/history
- **Reject when:** ambiguous order/future semantics
- **Validation:** time-aware validation
- **Inference state:** history/order state
- **Mandatory tests:** no future; group isolation

## 59. `shift_lead_rows`

- **Family:** temporal
- **Input:** value,time[,group]
- **Output:** same as value
- **Purpose:** future observation diagnostic
- **Preconditions:** strict order
- **Fit state:** none
- **Transform:** output value i+k
- **Null / invalid policy:** tail missing
- **Leakage class:** L5
- **Cost shape:** O(n)
- **Generate when:** diagnostic only
- **Reject when:** always as ordinary predictive feature
- **Validation:** never predictive
- **Inference state:** not allowed
- **Mandatory tests:** prove rejection

## 60. `lag_time_offset`

- **Family:** temporal
- **Input:** value,time[,group]
- **Output:** same as value
- **Purpose:** prior value at fixed time offset
- **Preconditions:** ordered timestamps
- **Fit state:** history index
- **Transform:** for row T query latest record <= T-delta
- **Null / invalid policy:** no history -> missing
- **Leakage class:** L3
- **Cost shape:** O(n log n) or merge-asof O(n)
- **Generate when:** irregular time observations
- **Reject when:** future nearest match/ambiguous group
- **Validation:** backtest/time-aware
- **Inference state:** history index/state
- **Mandatory tests:** <= query time only

## 61. `rolling_mean`

- **Family:** temporal
- **Input:** numeric,time[,group]
- **Output:** numeric
- **Purpose:** historical local level
- **Preconditions:** window and ordering defined
- **Fit state:** stateful history
- **Transform:** mean over prior window [T-W,T) by default
- **Null / invalid policy:** min_periods policy
- **Leakage class:** L3
- **Cost shape:** O(n) with deque/prefix state
- **Generate when:** sequence history
- **Reject when:** current/future inclusion
- **Validation:** time-aware
- **Inference state:** persist history
- **Mandatory tests:** insert-after-emit

## 62. `rolling_std`

- **Family:** temporal
- **Input:** numeric,time[,group]
- **Output:** numeric
- **Purpose:** historical volatility
- **Preconditions:** ordered window
- **Fit state:** stateful
- **Transform:** std over prior window
- **Null / invalid policy:** min_periods
- **Leakage class:** L3
- **Cost shape:** O(n)
- **Generate when:** volatility
- **Reject when:** tiny unstable window
- **Validation:** time-aware
- **Inference state:** history state
- **Mandatory tests:** future exclusion

## 63. `rolling_min`

- **Family:** temporal
- **Input:** numeric,time[,group]
- **Output:** numeric
- **Purpose:** historical lower envelope
- **Preconditions:** ordered window
- **Fit state:** stateful
- **Transform:** window min with monotonic deque
- **Null / invalid policy:** min_periods
- **Leakage class:** L3
- **Cost shape:** O(n)
- **Generate when:** range/bottom history
- **Reject when:** future/current leak
- **Validation:** time-aware
- **Inference state:** history state
- **Mandatory tests:** deque correctness

## 64. `rolling_max`

- **Family:** temporal
- **Input:** numeric,time[,group]
- **Output:** numeric
- **Purpose:** historical upper envelope
- **Preconditions:** ordered window
- **Fit state:** stateful
- **Transform:** window max with monotonic deque
- **Null / invalid policy:** min_periods
- **Leakage class:** L3
- **Cost shape:** O(n)
- **Generate when:** range/top history
- **Reject when:** future/current leak
- **Validation:** time-aware
- **Inference state:** history state
- **Mandatory tests:** deque correctness

## 65. `rolling_sum`

- **Family:** temporal
- **Input:** numeric,time[,group]
- **Output:** numeric
- **Purpose:** historical accumulated amount
- **Preconditions:** ordered window
- **Fit state:** stateful
- **Transform:** sum prior window
- **Null / invalid policy:** min_periods/0 policy
- **Leakage class:** L3
- **Cost shape:** O(n)
- **Generate when:** recent spend/usage
- **Reject when:** future/current inclusion
- **Validation:** time-aware
- **Inference state:** history state
- **Mandatory tests:** boundary

## 66. `rolling_count`

- **Family:** temporal
- **Input:** event,time[,group]
- **Output:** numeric
- **Purpose:** recent event frequency
- **Preconditions:** ordered window
- **Fit state:** stateful
- **Transform:** count events in prior window
- **Null / invalid policy:** 0 if none
- **Leakage class:** L3
- **Cost shape:** O(n)
- **Generate when:** activity frequency
- **Reject when:** wrong interval closure
- **Validation:** time-aware
- **Inference state:** history state
- **Mandatory tests:** boundary

## 67. `rolling_unique`

- **Family:** temporal
- **Input:** categorical,time[,group]
- **Output:** numeric
- **Purpose:** recent diversity
- **Preconditions:** ordered window
- **Fit state:** stateful multiset counter
- **Transform:** distinct count in prior window
- **Null / invalid policy:** 0/min_periods
- **Leakage class:** L3
- **Cost shape:** O(n) expected, memory by cardinality
- **Generate when:** behavior diversity
- **Reject when:** memory cap
- **Validation:** time-aware
- **Inference state:** history state
- **Mandatory tests:** counter eviction

## 68. `expanding_mean`

- **Family:** temporal
- **Input:** numeric,time[,group]
- **Output:** numeric
- **Purpose:** long-term historical mean
- **Preconditions:** ordered history
- **Fit state:** stateful
- **Transform:** mean of all prior available observations
- **Null / invalid policy:** first missing/prior
- **Leakage class:** L3
- **Cost shape:** O(n)
- **Generate when:** long-term baseline
- **Reject when:** current included before prediction
- **Validation:** time-aware
- **Inference state:** running stats
- **Mandatory tests:** insert-after-emit

## 69. `ewma`

- **Family:** temporal
- **Input:** numeric,time[,group]
- **Output:** numeric
- **Purpose:** recency-weighted history
- **Preconditions:** ordered history; alpha
- **Fit state:** stateful
- **Transform:** s_t=alpha*x_prev+(1-alpha)*s_prev
- **Null / invalid policy:** initialization policy
- **Leakage class:** L3
- **Cost shape:** O(n)
- **Generate when:** recent history important
- **Reject when:** alpha search too broad/no gain
- **Validation:** time-aware
- **Inference state:** persist EWMA state
- **Mandatory tests:** online=batch equivalence

## 70. `time_since_previous`

- **Family:** temporal
- **Input:** time[,group]
- **Output:** numeric
- **Purpose:** inter-event gap
- **Preconditions:** ordered timestamps
- **Fit state:** stateful/none batch
- **Transform:** T_i - T_{i-1}
- **Null / invalid policy:** first missing
- **Leakage class:** L3
- **Cost shape:** O(n log n) sort
- **Generate when:** event cadence
- **Reject when:** duplicate time ambiguous
- **Validation:** time-aware
- **Inference state:** previous time state
- **Mandatory tests:** non-negative gaps

## 71. `time_since_first`

- **Family:** temporal
- **Input:** time,group
- **Output:** numeric
- **Purpose:** entity age since first seen
- **Preconditions:** group history
- **Fit state:** stateful
- **Transform:** T-current group first_time
- **Null / invalid policy:** first row 0
- **Leakage class:** L3
- **Cost shape:** O(n) after sort
- **Generate when:** entity lifecycle
- **Reject when:** first time learned from future
- **Validation:** time-aware
- **Inference state:** persist first_time
- **Mandatory tests:** no future first

## 72. `events_since_last_flag`

- **Family:** temporal
- **Input:** flag,time,group
- **Output:** numeric
- **Purpose:** time since prior flagged event
- **Preconditions:** ordered history
- **Fit state:** stateful
- **Transform:** T - last T where flag true and available
- **Null / invalid policy:** no prior -> missing
- **Leakage class:** L3
- **Cost shape:** O(n)
- **Generate when:** recency of action/event
- **Reject when:** current flag post-outcome
- **Validation:** time-aware
- **Inference state:** last-event state
- **Mandatory tests:** availability

## 73. `group_count`

- **Family:** group aggregate
- **Input:** group key,value
- **Output:** numeric
- **Purpose:** group support
- **Preconditions:** repeated meaningful groups
- **Fit state:** learn map on training fold/history
- **Transform:** count rows per key then map to rows
- **Null / invalid policy:** unknown group -> global prior/missing/0 policy
- **Leakage class:** L1 or L3 temporal
- **Cost shape:** O(n) + group state
- **Generate when:** repeated entity/group structure
- **Reject when:** mostly unique group, memory/leakage risk
- **Validation:** inside-fold or time-aware
- **Inference state:** store aggregate map/state
- **Mandatory tests:** unknown group; fold isolation

## 74. `group_mean`

- **Family:** group aggregate
- **Input:** group key,value
- **Output:** numeric
- **Purpose:** group mean
- **Preconditions:** repeated meaningful groups
- **Fit state:** learn map on training fold/history
- **Transform:** mean value per key then map to rows
- **Null / invalid policy:** unknown group -> global prior/missing/0 policy
- **Leakage class:** L1 or L3 temporal
- **Cost shape:** O(n) + group state
- **Generate when:** repeated entity/group structure
- **Reject when:** mostly unique group, memory/leakage risk
- **Validation:** inside-fold or time-aware
- **Inference state:** store aggregate map/state
- **Mandatory tests:** unknown group; fold isolation

## 75. `group_std`

- **Family:** group aggregate
- **Input:** group key,value
- **Output:** numeric
- **Purpose:** group dispersion
- **Preconditions:** repeated meaningful groups
- **Fit state:** learn map on training fold/history
- **Transform:** standard deviation per key then map to rows
- **Null / invalid policy:** unknown group -> global prior/missing/0 policy
- **Leakage class:** L1 or L3 temporal
- **Cost shape:** O(n) + group state
- **Generate when:** repeated entity/group structure
- **Reject when:** mostly unique group, memory/leakage risk
- **Validation:** inside-fold or time-aware
- **Inference state:** store aggregate map/state
- **Mandatory tests:** unknown group; fold isolation

## 76. `group_min`

- **Family:** group aggregate
- **Input:** group key,value
- **Output:** numeric
- **Purpose:** group minimum
- **Preconditions:** repeated meaningful groups
- **Fit state:** learn map on training fold/history
- **Transform:** minimum value per key then map to rows
- **Null / invalid policy:** unknown group -> global prior/missing/0 policy
- **Leakage class:** L1 or L3 temporal
- **Cost shape:** O(n) + group state
- **Generate when:** repeated entity/group structure
- **Reject when:** mostly unique group, memory/leakage risk
- **Validation:** inside-fold or time-aware
- **Inference state:** store aggregate map/state
- **Mandatory tests:** unknown group; fold isolation

## 77. `group_max`

- **Family:** group aggregate
- **Input:** group key,value
- **Output:** numeric
- **Purpose:** group maximum
- **Preconditions:** repeated meaningful groups
- **Fit state:** learn map on training fold/history
- **Transform:** maximum value per key then map to rows
- **Null / invalid policy:** unknown group -> global prior/missing/0 policy
- **Leakage class:** L1 or L3 temporal
- **Cost shape:** O(n) + group state
- **Generate when:** repeated entity/group structure
- **Reject when:** mostly unique group, memory/leakage risk
- **Validation:** inside-fold or time-aware
- **Inference state:** store aggregate map/state
- **Mandatory tests:** unknown group; fold isolation

## 78. `group_median`

- **Family:** group aggregate
- **Input:** group key,value
- **Output:** numeric
- **Purpose:** group robust center
- **Preconditions:** repeated meaningful groups
- **Fit state:** learn map on training fold/history
- **Transform:** median per key then map to rows
- **Null / invalid policy:** unknown group -> global prior/missing/0 policy
- **Leakage class:** L1 or L3 temporal
- **Cost shape:** O(n) + group state
- **Generate when:** repeated entity/group structure
- **Reject when:** mostly unique group, memory/leakage risk
- **Validation:** inside-fold or time-aware
- **Inference state:** store aggregate map/state
- **Mandatory tests:** unknown group; fold isolation

## 79. `group_nunique`

- **Family:** group aggregate
- **Input:** group key,value
- **Output:** numeric
- **Purpose:** group diversity
- **Preconditions:** repeated meaningful groups
- **Fit state:** learn map on training fold/history
- **Transform:** distinct value count per key then map to rows
- **Null / invalid policy:** unknown group -> global prior/missing/0 policy
- **Leakage class:** L1 or L3 temporal
- **Cost shape:** O(n) + group state
- **Generate when:** repeated entity/group structure
- **Reject when:** mostly unique group, memory/leakage risk
- **Validation:** inside-fold or time-aware
- **Inference state:** store aggregate map/state
- **Mandatory tests:** unknown group; fold isolation

## 80. `group_missing_rate`

- **Family:** group aggregate
- **Input:** group key,value
- **Output:** numeric
- **Purpose:** group data-quality pattern
- **Preconditions:** repeated meaningful groups
- **Fit state:** learn map on training fold/history
- **Transform:** missing fraction per key then map to rows
- **Null / invalid policy:** unknown group -> global prior/missing/0 policy
- **Leakage class:** L1 or L3 temporal
- **Cost shape:** O(n) + group state
- **Generate when:** repeated entity/group structure
- **Reject when:** mostly unique group, memory/leakage risk
- **Validation:** inside-fold or time-aware
- **Inference state:** store aggregate map/state
- **Mandatory tests:** unknown group; fold isolation

## 81. `text_length_chars`

- **Family:** text
- **Input:** text
- **Output:** numeric
- **Purpose:** character length
- **Preconditions:** text
- **Fit state:** none
- **Transform:** len(text)
- **Null / invalid policy:** missing -> 0 plus missing indicator optional
- **Leakage class:** L0
- **Cost shape:** O(total chars)
- **Generate when:** text exists
- **Reject when:** no variation
- **Validation:** incremental CV
- **Inference state:** text available
- **Mandatory tests:** unicode length policy

## 82. `text_length_tokens`

- **Family:** text
- **Input:** text
- **Output:** numeric
- **Purpose:** token count
- **Preconditions:** text and tokenizer policy
- **Fit state:** fixed tokenizer config or learned tokenizer state
- **Transform:** tokenize and count
- **Null / invalid policy:** missing -> 0
- **Leakage class:** L0/L1
- **Cost shape:** O(total chars)
- **Generate when:** text exists
- **Reject when:** cost/no variation
- **Validation:** incremental CV
- **Inference state:** same tokenizer
- **Mandatory tests:** tokenizer reproducibility

## 83. `text_digit_ratio`

- **Family:** text
- **Input:** text
- **Output:** numeric
- **Purpose:** numeric-character intensity
- **Preconditions:** text
- **Fit state:** none
- **Transform:** digits/max(chars,1)
- **Null / invalid policy:** empty -> 0
- **Leakage class:** L0
- **Cost shape:** O(total chars)
- **Generate when:** codes/messages
- **Reject when:** no variation
- **Validation:** incremental CV
- **Inference state:** text available
- **Mandatory tests:** empty

## 84. `text_upper_ratio`

- **Family:** text
- **Input:** text
- **Output:** numeric
- **Purpose:** uppercase intensity
- **Preconditions:** text
- **Fit state:** none
- **Transform:** uppercase letters/max(letters,1)
- **Null / invalid policy:** no letters -> 0
- **Leakage class:** L0
- **Cost shape:** O(total chars)
- **Generate when:** style signal
- **Reject when:** no variation
- **Validation:** incremental CV
- **Inference state:** text available
- **Mandatory tests:** unicode case

## 85. `text_punctuation_ratio`

- **Family:** text
- **Input:** text
- **Output:** numeric
- **Purpose:** punctuation intensity
- **Preconditions:** text
- **Fit state:** none
- **Transform:** punctuation/max(chars,1)
- **Null / invalid policy:** empty -> 0
- **Leakage class:** L0
- **Cost shape:** O(total chars)
- **Generate when:** messages/logs
- **Reject when:** no variation
- **Validation:** incremental CV
- **Inference state:** text available
- **Mandatory tests:** unicode punctuation

## 86. `tfidf_word`

- **Family:** text representation
- **Input:** text
- **Output:** sparse numeric
- **Purpose:** word bag-of-words representation
- **Preconditions:** enough text/support
- **Fit state:** fit vocabulary and document frequencies on training
- **Transform:** TF-IDF with normalization
- **Null / invalid policy:** unknown tokens ignored
- **Leakage class:** L1
- **Cost shape:** O(tokens) + sparse matrix
- **Generate when:** predictive text and memory allows
- **Reject when:** vocab too large/memory limit
- **Validation:** inside-fold
- **Inference state:** store vocab/idf
- **Mandatory tests:** fold-only vocab

## 87. `tfidf_char`

- **Family:** text representation
- **Input:** text
- **Output:** sparse numeric
- **Purpose:** subword representation
- **Preconditions:** enough text
- **Fit state:** fit char-ngram vocab/idf
- **Transform:** char ngram TF-IDF
- **Null / invalid policy:** unknown ngrams ignored
- **Leakage class:** L1
- **Cost shape:** high
- **Generate when:** misspellings/codes
- **Reject when:** cost/memory
- **Validation:** inside-fold
- **Inference state:** store vocab/idf
- **Mandatory tests:** boundary markers

## 88. `text_hashing`

- **Family:** text representation
- **Input:** text
- **Output:** sparse numeric
- **Purpose:** bounded-memory token features
- **Preconditions:** hash dimension fixed
- **Fit state:** fixed tokenizer/hash seed
- **Transform:** hash tokens/ngrams to buckets
- **Null / invalid policy:** unknown naturally hashed
- **Leakage class:** L0
- **Cost shape:** O(tokens)
- **Generate when:** very large vocab
- **Reject when:** collision/interpretability
- **Validation:** incremental CV
- **Inference state:** same seed/dim
- **Mandatory tests:** determinism

## 89. `text_svd`

- **Family:** dimensionality
- **Input:** TFIDF sparse
- **Output:** dense numeric
- **Purpose:** compress sparse text
- **Preconditions:** TFIDF exists, k budgeted
- **Fit state:** fit truncated SVD on training
- **Transform:** project to k components
- **Null / invalid policy:** empty -> zeros
- **Leakage class:** L1
- **Cost shape:** O(nnz*k)
- **Generate when:** large sparse text + dense learner
- **Reject when:** small n/cost/no gain
- **Validation:** inside-fold
- **Inference state:** store components
- **Mandatory tests:** projection equivalence

## 90. `pca`

- **Family:** dimensionality
- **Input:** numeric matrix
- **Output:** dense numeric
- **Purpose:** compress correlated numeric dimensions
- **Preconditions:** scaled/imputed numeric; n,p sufficient
- **Fit state:** fit mean/components training only
- **Transform:** center/scale and project
- **Null / invalid policy:** preprocess missing
- **Leakage class:** L1
- **Cost shape:** O(np*k)
- **Generate when:** high-dimensional correlated numeric
- **Reject when:** interpretability/no need/cost
- **Validation:** inside-fold
- **Inference state:** store preprocessing/components
- **Mandatory tests:** component shape

## 91. `truncated_svd`

- **Family:** dimensionality
- **Input:** sparse matrix
- **Output:** dense numeric
- **Purpose:** compress sparse representations
- **Preconditions:** sparse input; k feasible
- **Fit state:** fit components training only
- **Transform:** matrix projection
- **Null / invalid policy:** sparse zeros natural
- **Leakage class:** L1
- **Cost shape:** O(nnz*k)
- **Generate when:** TFIDF/high-dimensional sparse
- **Reject when:** k too high/cost
- **Validation:** inside-fold
- **Inference state:** store components
- **Mandatory tests:** shape

## 92. `kmeans_label`

- **Family:** cluster
- **Input:** numeric matrix
- **Output:** categorical
- **Purpose:** coarse regime membership
- **Preconditions:** scaled/imputed
- **Fit state:** fit centers training only
- **Transform:** nearest center id
- **Null / invalid policy:** preprocess missing
- **Leakage class:** L1
- **Cost shape:** O(nkp)
- **Generate when:** cluster structure prior
- **Reject when:** unstable/no gain/cost
- **Validation:** inside-fold
- **Inference state:** store centers/preprocess
- **Mandatory tests:** cluster permutation stability

## 93. `kmeans_distances`

- **Family:** cluster
- **Input:** numeric matrix
- **Output:** numeric vector
- **Purpose:** distance to prototypes
- **Preconditions:** scaled/imputed
- **Fit state:** fit centers training only
- **Transform:** distance to each center
- **Null / invalid policy:** preprocess missing
- **Leakage class:** L1
- **Cost shape:** O(nkp)
- **Generate when:** cluster geometry
- **Reject when:** too many centers/latency
- **Validation:** inside-fold
- **Inference state:** store centers
- **Mandatory tests:** distance finite

## 94. `residual_interaction_proposal`

- **Family:** model-informed
- **Input:** OOF residuals + source features
- **Output:** candidate operation
- **Purpose:** propose interactions where baseline errors remain structured
- **Preconditions:** valid OOF residuals only
- **Fit state:** proposal model/statistics on development only
- **Transform:** rank pair/transform hypotheses; child op defines actual transform
- **Null / invalid policy:** child semantics
- **Leakage class:** L2 proposal
- **Cost shape:** variable
- **Generate when:** stable residual slices
- **Reject when:** uses final holdout/in-sample residuals
- **Validation:** nested development evaluation
- **Inference state:** store proposal evidence
- **Mandatory tests:** OOF-only

## 95. `tree_leaf_oof`

- **Family:** model-informed
- **Input:** tree model + features
- **Output:** categorical/sparse
- **Purpose:** use tree partitions as representation
- **Preconditions:** tree candidate exists
- **Fit state:** fit tree per training fold
- **Transform:** OOF leaf indices then encode/hash
- **Null / invalid policy:** n/a
- **Leakage class:** L2
- **Cost shape:** high
- **Generate when:** stacking/representation evidence
- **Reject when:** overfit/latency/memory
- **Validation:** OOF only for downstream training
- **Inference state:** store final tree
- **Mandatory tests:** in-sample prohibition


# Exact shift / lag / rolling procedure

A simple dataframe `.shift()` call is not enough. FIAE must prove ordering and prediction-time availability.

## Row-based lag `shift(X, k)`

### Inputs
- value column `X`;
- event/order column `T`;
- optional entity/group `G`;
- positive integer lag `k`;
- optional deterministic tie-break key;
- prediction-time semantics.

### Steps

1. Reject `k <= 0` for a predictive lag.
2. Require an ordering key. File row order is not automatically meaningful.
3. If group key exists, partition history by group.
4. Stable-sort each group by `(T, tie_breaker)`.
5. If duplicate times exist with no deterministic tie-break semantics, emit `AMBIGUOUS_ORDER`.
6. For sorted position `i`, candidate value is `X[i-k]` if `i>=k`.
7. First `k` rows per group are missing.
8. Never backfill missing lag values from a later row.
9. Restore original row order using preserved row identity.
10. Record group/order/tie/lag/boundary in lineage.
11. Validate with time-aware splitting.
12. If `X` is the target, also prove the old label was available before prediction time.
13. At inference, maintain the same history state or query an equivalent historical store.
14. Unit test group boundaries.
15. Property test that output for time `T` never depends on source time `>T`.

Pseudocode:

```python
def lag_rows(rows, *, value, time, k, group=None, tie=None):
    if k < 1:
        raise InvalidLag(k)
    ordered = stable_sort_with_original_index(rows, [group, time, tie])
    out = missing_array(len(rows))
    for partition in partition_by(ordered, group):
        for i, row in enumerate(partition):
            if i >= k:
                out[row.original_index] = partition[i-k][value]
    return out
```

## Time-offset lag

For irregular observations, `k rows ago` may not equal `7 days ago`.

Steps:
1. sort by group/time;
2. for each prediction at `T`, compute `Q=T-delta`;
3. select the latest history record with timestamp `<=Q`;
4. never select nearest absolute timestamp if it could be future;
5. no match -> missing;
6. implement efficiently with merge-as-of/two pointers or indexed binary search;
7. validate in chronological backtests.

## Rolling window `[T-W, T)`

Default interval excludes the current observation.

For each group:
1. sort history;
2. initialize deque/counter/running aggregate;
3. before scoring current row, evict history older than lower boundary;
4. compute aggregate on current state;
5. emit feature;
6. only after emission, insert current row if it becomes available for future predictions.

The **insert-after-emit** ordering prevents using the current target/value when the feature is intended to describe prior history.

## Target-derived lag/rolling

Define label availability timestamp `A(Y_t)`.

A historical target value may enter feature state only if:

```text
A(Y_t) <= prediction_time_current
```

If label delay is unknown, target-history features are review-required and disabled by default.

## Temporal tests

- first-k missing;
- no cross-group contamination;
- shuffled source order yields same result after semantic sort;
- duplicate time ambiguity;
- future timestamp never referenced;
- batch transform equals online stateful transform;
- label-delay safety;
- window boundary inclusion/exclusion;
- no backfill from future.

## Research basis

- [H2O Driverless AI feature engineering](https://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/feature-engineering.html)
- [DataRobot modeling process](https://docs.datarobot.com/latest/en/docs/reference/pred-ai-ref/model-ref.html)
- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
