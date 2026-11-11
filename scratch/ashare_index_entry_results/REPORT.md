# CSI300 index-entry independent-source Alpha experiment

- Decision: `retain_research_only`
- Selected candidate: `positive_pre_runup`
- Factor ID: `CSI300_ENTRY_RUNUP_EXHAUSTION_REVERSAL20`
- Independent alpha: `False`
- Source SHA-256: `e412f4bfe27c7f4d789f3c9d25355fb3690bdb0ca5cbaa4fc059161003139cea`

## Frozen gates

- `selection_candidate_exists`: `True`
- `confirmation_support`: `True`
- `final_support`: `True`
- `confirmation_target_negative`: `True`
- `final_target_negative`: `True`
- `confirmation_parent_residual_negative`: `False`
- `final_parent_residual_negative`: `True`
- `confirmation_complement_difference_negative`: `False`
- `final_complement_difference_negative`: `True`
- `target_block1_upper_negative`: `True`
- `target_block2_upper_negative`: `True`
- `residual_block1_upper_negative`: `False`
- `residual_block2_upper_negative`: `False`
- `complement_block1_upper_negative`: `False`
- `complement_block2_upper_negative`: `False`
- `all_post_selection_years_negative`: `True`
- `final_overlay_mse_positive`: `False`

## Selected candidate metrics

| candidate          | period       |   n_events |   n_batches |   event_mean |   event_median |   negative_share |   batch_equal_mean |   batch_t |   batch_p_one_sided |   parent_residual_mean |   candidate_minus_complement |
|:-------------------|:-------------|-----------:|------------:|-------------:|---------------:|-----------------:|-------------------:|----------:|--------------------:|-----------------------:|-----------------------------:|
| positive_pre_runup | development  |         27 |           2 |  -0.0218454  |     -0.02897   |         0.555556 |        -0.0232812  | -0.60056  |            0.327848 |          nan           |                    0.0206044 |
| positive_pre_runup | selection    |         31 |           2 |  -0.0166609  |     -0.022312  |         0.580645 |        -0.0377982  | -0.749899 |            0.295188 |           -0.0939233   |                   -0.0531059 |
| positive_pre_runup | confirmation |         14 |           2 |  -0.00422921 |     -0.0024935 |         0.5      |        -0.00422921 | -1.10943  |            0.233502 |            0.00958026  |                    0.0168721 |
| positive_pre_runup | final        |         10 |           2 |  -0.0241382  |     -0.015896  |         0.8      |        -0.0220502  | -2.11205  |            0.140757 |           -0.000950061 |                   -0.0024375 |

## Intervals

```json
{
  "target_block1_batches": [
    -0.013139690476190485,
    -0.026378071428571265,
    -0.003215357142857268
  ],
  "target_block2_batches": [
    -0.013139690476190485,
    -0.022050166666666586,
    -0.0042292142857143855
  ],
  "parent_residual_block1_batches": [
    0.005247699535783931,
    -0.022036631555523774,
    0.03686458741862263
  ],
  "parent_residual_block2_batches": [
    0.005247699535783931,
    0.0009151427442529421,
    0.031098784254520156
  ],
  "candidate_minus_complement_block1_batches": [
    0.007217281746031727,
    -0.013646626984127011,
    0.03268636904761907
  ],
  "candidate_minus_complement_block2_batches": [
    0.007217281746031727,
    -0.013646626984127011,
    0.016872063492063463
  ],
  "calendar_year_means": [
    -0.0042292142857143855,
    -0.024138199999999888
  ]
}
```

## Parent overlay

```json
{
  "n_events": 10,
  "baseline_mse": 0.0024015241292014208,
  "overlay_mse": 0.01104464747157808,
  "mse_improvement": -0.008643123342376659,
  "baseline_bias": -0.0009500607739532842,
  "overlay_bias": 0.09297325399194581
}
```

## Evidence boundary

The index-membership event is an external information source. Passing these gates would admit only a default-off historical candidate. It would not certify execution, costs, borrow, capacity, a continuous account, production use, or fresh future-market confirmation.
