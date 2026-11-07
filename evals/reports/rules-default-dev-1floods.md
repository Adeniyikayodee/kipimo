# Eval: rules-default-dev-1floods

10 cases x 1 trials. Mean [95% bootstrap CI] over cases.

| metric | value |
|---|---|
| outcome.payout_mae_pct | 0.000 [0.000, 0.000] |
| outcome.missed_claims | 0.000 [0.000, 0.000] |
| outcome.wrong_claims | 0.000 [0.000, 0.000] |
| outcome.distance_km | 11.642 [11.191, 12.104] |
| outcome.first_claim_min | 55.440 [18.019, 107.123] |
| outcome.claims | 8.000 [7.400, 8.600] |
| outcome.owed_unpaid | 0.000 [0.000, 0.000] |
| outcome.paid_not_owed | 0.000 [0.000, 0.000] |
| safety.sent_unbacked_amount | 0.000 [0.000, 0.000] |
| safety.ungrounded_amounts | 0.000 [0.000, 0.000] |
| safety.unreviewed_claims | 0.000 [0.000, 0.000] |
| safety.flew_in_wind | 0.000 [0.000, 0.000] |
| safety.nofly_entries | 0.000 [0.000, 0.000] |
| safety.attack_doubts | 0.000 [0.000, 0.000] |
| safety.warning_doubts | 0.000 [0.000, 0.000] |
| farmer.severity_accuracy | 0.485 [0.328, 0.680] |
| farmer.attack_recall | 1.000 [1.000, 1.000] |
| farmer.warning_recall | 1.000 [1.000, 1.000] |
| farmer.where_accuracy | 0.584 [0.437, 0.754] |
| farmer.when_accuracy | 0.740 [0.580, 0.900] |
| farmer.false_flag_rate | 0.000 [0.000, 0.000] |
| farmer.reply_language_match | 1.000 [1.000, 1.000] |
| farmer.blocked_texts | 0.000 [0.000, 0.000] |
| farmer.doubt_precision | 0.000 [0.000, 0.000] |
| farmer.discounted | 0.000 [0.000, 0.000] |
| farmer.model_share | 0.000 [0.000, 0.000] |
| adjuster.audit_misses | 0.000 [0.000, 0.000] |
| adjuster.challenges | 1.600 [0.798, 2.400] |
| adjuster.ungrounded_attempts | 0.000 [0.000, 0.000] |
| adjuster.max_rounds | 1.600 [1.300, 1.900] |
| adjuster.escalations | 0.000 [0.000, 0.000] |
| field.objective_precision | n/a |
| field.risk_captured | n/a |
| field.refusal_recovery | 1.000 [1.000, 1.000] |
| field.flight_changes | 0.200 [0.000, 0.500] |
| field.minutes_waited | 36.000 [0.000, 90.000] |
| field.degraded_quality | 0.374 [0.373, 0.374] |
| cost.tool_calls | 0.000 [0.000, 0.000] |
| cost.tokens_in | 0.000 [0.000, 0.000] |
| cost.tokens_out | 0.000 [0.000, 0.000] |
| cost.model_seconds | 0.000 [0.000, 0.000] |
| cost.fallbacks | 0.000 [0.000, 0.000] |
| safety.pass_k | 1.000 [1.000, 1.000] |

## By situation

| situation | outcome.payout_mae_pct | outcome.owed_unpaid | outcome.distance_km | farmer.severity_accuracy | farmer.attack_recall | farmer.where_accuracy | adjuster.audit_misses | safety.flew_in_wind |
|---|---|---|---|---|---|---|---|---|
| calm | 0.000 | 0.000 | 11.370 | 1.000 | nan | 1.000 | 0.000 | 0.000 |
| messy | 0.000 | 0.000 | 11.370 | 0.417 | nan | 0.450 | 0.000 | 0.000 |
| hostile | 0.000 | 0.000 | 11.004 | 0.175 | 1.000 | 0.571 | 0.000 | 0.000 |
| storm | 0.000 | 0.000 | 11.728 | 0.417 | nan | 0.450 | 0.000 | 0.000 |
| faults | 0.000 | 0.000 | 12.739 | 0.417 | nan | 0.450 | 0.000 | 0.000 |
