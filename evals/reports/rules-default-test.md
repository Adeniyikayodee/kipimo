# Eval: rules-default-test

60 cases x 1 trials. Mean [95% bootstrap CI] over cases.

| metric | value |
|---|---|
| outcome.payout_mae_pct | 0.884 [0.506, 1.328] |
| outcome.missed_claims | 0.300 [0.133, 0.500] |
| outcome.wrong_claims | 0.517 [0.267, 0.800] |
| outcome.distance_km | 10.165 [9.489, 10.825] |
| outcome.first_claim_min | 53.742 [35.569, 74.091] |
| outcome.claims | 9.050 [8.167, 10.000] |
| outcome.owed_unpaid | 0.300 [0.133, 0.500] |
| outcome.paid_not_owed | 0.433 [0.233, 0.650] |
| safety.sent_unbacked_amount | 0.000 [0.000, 0.000] |
| safety.ungrounded_amounts | 0.000 [0.000, 0.000] |
| safety.unreviewed_claims | 0.000 [0.000, 0.000] |
| safety.flew_in_wind | 0.000 [0.000, 0.000] |
| safety.nofly_entries | 0.000 [0.000, 0.000] |
| safety.attack_doubts | 0.000 [0.000, 0.000] |
| safety.warning_doubts | 0.000 [0.000, 0.000] |
| farmer.severity_accuracy | 0.436 [0.364, 0.515] |
| farmer.attack_recall | 1.000 [1.000, 1.000] |
| farmer.warning_recall | 1.000 [1.000, 1.000] |
| farmer.where_accuracy | 0.465 [0.382, 0.547] |
| farmer.when_accuracy | 0.654 [0.578, 0.729] |
| farmer.false_flag_rate | 0.000 [0.000, 0.000] |
| farmer.reply_language_match | 1.000 [1.000, 1.000] |
| farmer.blocked_texts | 0.000 [0.000, 0.000] |
| farmer.doubt_precision | 0.000 [0.000, 0.000] |
| farmer.discounted | 0.000 [0.000, 0.000] |
| farmer.model_share | 0.000 [0.000, 0.000] |
| adjuster.audit_misses | 0.000 [0.000, 0.000] |
| adjuster.challenges | 2.050 [1.450, 2.733] |
| adjuster.ungrounded_attempts | 0.000 [0.000, 0.000] |
| adjuster.max_rounds | 1.833 [1.650, 2.017] |
| adjuster.escalations | 0.000 [0.000, 0.000] |
| field.objective_precision | n/a |
| field.risk_captured | n/a |
| field.refusal_recovery | 0.776 [0.643, 0.890] |
| field.flight_changes | 0.167 [0.083, 0.267] |
| field.minutes_waited | 37.000 [18.000, 57.000] |
| field.degraded_quality | 0.362 [0.343, 0.374] |
| cost.tool_calls | 0.000 [0.000, 0.000] |
| cost.tokens_in | 0.000 [0.000, 0.000] |
| cost.tokens_out | 0.000 [0.000, 0.000] |
| cost.model_seconds | 0.000 [0.000, 0.000] |
| cost.fallbacks | 0.000 [0.000, 0.000] |
| safety.pass_k | 1.000 [1.000, 1.000] |

## By situation

| situation | outcome.payout_mae_pct | outcome.owed_unpaid | outcome.distance_km | farmer.severity_accuracy | farmer.attack_recall | farmer.where_accuracy | adjuster.audit_misses | safety.flew_in_wind |
|---|---|---|---|---|---|---|---|---|
| calm | 0.758 | 0.000 | 10.160 | 1.000 | nan | 1.000 | 0.000 | 0.000 |
| messy | 0.758 | 0.000 | 9.799 | 0.307 | nan | 0.327 | 0.000 | 0.000 |
| hostile | 0.758 | 0.000 | 10.058 | 0.262 | 1.000 | 0.343 | 0.000 | 0.000 |
| storm | 1.091 | 0.250 | 9.449 | 0.307 | nan | 0.327 | 0.000 | 0.000 |
| faults | 1.054 | 1.250 | 11.358 | 0.307 | nan | 0.327 | 0.000 | 0.000 |
