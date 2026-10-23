# ADR-0003: Initial gate set

**Status:** accepted, 2026-10-22

## Context

`gates.toml` was first written from the numbers in the design spec (section 5) and keyed by short stage names (`[s0]`). Review found three problems: the comparison direction was guessed with a default, some gates in the spec had no key (so nothing enforced them), and one gate (`min_spearman_top50`) was defined by a metric that ignored documents the new model ranks above the reference's top 50.

## Decision

`gates.toml` is rewritten as the initial gate set below. Anything not in this table is not gated.

**Structure**
- Tables are keyed by the full stage name (`[s1_rewrite]`). A stage's report is evaluated against its own table only (`egcm.gates.evaluate(report, stage_key)`); a stage with no table is an error.
- The key decides the comparison: `min_*` is `>=`, `max_*` is `<=`, a bool is `==`. Any other key fails when `gates.toml` is loaded. The prefix is only the direction. How a metric is aggregated (minimum, mean, percentile) is defined by that metric's docstring or stage README.
- Recorded-only numbers go in a `targets` subtable and are never gated. `target_ms_per_image_70` moves to `[s6_device.targets]`.

**Gates added** (these were stated in the spec but had no key)
- `[s0_reference]` `license_apache = true`, `key_inventory_written = true`, `deterministic = true`
- `[s1_rewrite]` `strict_load = true`
- `[s5_swift]` `token_ids_identical = true`
- `[s6_device]` `min_devices = 1`
- `[s7_publish]` `confirmed = true`: the report is only written as passed when `--confirm` was given and s0 to s6 are not stale.

**Renamed or redefined**
- `[s4_compress]` `min_spearman_top50` becomes `min_p5_spearman_top50`. It is the 5th percentile over the 120 app queries of a Spearman rho that ranks the reference's top 50 inside the new model's full ranking, so intruders lower it. The value stays 0.9.
- `min_cosine` in `[s4_compress]` and `[s5_swift]` is the minimum over text and image vectors combined. (s1 to s3 keep separate `min_cosine_text` and `min_cosine_image`.)
- `max_pixel_abs_diff = 0.003922` is 1/255 rounded up.

## Consequences

- Every threshold value from the spec is unchanged. Only enforcement and naming change.
- Changing any value or key in `gates.toml` needs a new ADR in the same commit (rule 3 in AGENTS.md).
