"""
Your defense. Implement register(ctx) and a handler per event type.
See ../README.md for the full interface + toolkit reference, and
../RULES.md before you start.

Strategy: purely static / deterministic thresholds derived from offline
simulation.  Adaptive z-score checks were removed because the clean-data
variance is large enough to produce unacceptable false-positive rates.
All thresholds are calibrated so that FPR = 0 on the practice schedule
while TPR remains maximised.
"""
from api import Verdict


# --- Helpers -----------------------------------------------------------------

def _budget_ok(ctx, cost):
    """True if there is enough budget remaining for this call."""
    return ctx.tools.budget_remaining() >= cost


# --- Registration ------------------------------------------------------------

def register(ctx):
    ctx.on("data_batch",              check_data_batch)
    ctx.on("contract_checkpoint",     check_contract_checkpoint)
    ctx.on("lineage_run",             check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch",         check_embedding_batch)


# --- Handlers ----------------------------------------------------------------

def check_data_batch(payload, ctx):
    if not _budget_ok(ctx, 1.0):
        return Verdict(alert=False, pillar="checks")

    profile = ctx.tools.batch_profile(payload["batch_id"])
    if "error" in profile:
        return Verdict(alert=False, pillar="checks")

    # Static baseline checks only (adaptive z-score removed: too many FPs)
    row_count = profile.get("row_count")
    if row_count is not None:
        if row_count < ctx.baseline["row_count_min"] or row_count > ctx.baseline["row_count_max"]:
            return Verdict(alert=True, pillar="checks", reason=f"row_count {row_count} out of bounds")

    null_rates = profile.get("null_rate", {})
    for col, rate in null_rates.items():
        if rate > ctx.baseline["null_rate_max"]:
            return Verdict(alert=True, pillar="checks", reason=f"null_rate {rate} on {col} exceeds baseline")

    mean_amount = profile.get("mean_amount")
    if mean_amount is not None:
        if mean_amount < ctx.baseline["mean_amount_min"] or mean_amount > ctx.baseline["mean_amount_max"]:
            return Verdict(alert=True, pillar="checks", reason=f"mean_amount {mean_amount} out of bounds")

    staleness_min = profile.get("staleness_min")
    if staleness_min is not None:
        if staleness_min > ctx.baseline["staleness_min_max"]:
            return Verdict(alert=True, pillar="checks", reason=f"staleness {staleness_min} exceeds baseline")

    return Verdict(alert=False, pillar="checks")


def check_contract_checkpoint(payload, ctx):
    if not _budget_ok(ctx, 1.5):
        return Verdict(alert=False, pillar="contracts")

    diff = ctx.tools.contract_diff(payload["contract_id"], payload["checkpoint_batch_id"])
    if "error" in diff:
        return Verdict(alert=False, pillar="contracts")

    # Schema / type violations are hard signals
    violations = diff.get("violations", [])
    if len(violations) > 0:
        return Verdict(alert=True, pillar="contracts", reason=f"contract violations: {violations}")

    freshness_delay = diff.get("freshness_delay_min")
    if freshness_delay is not None:
        # Per-event declared SLA (tightest bound)
        declared_sla = payload.get("declared_sla", {})
        sla_freshness = declared_sla.get("freshness_min")
        if sla_freshness is not None and freshness_delay > sla_freshness:
            return Verdict(alert=True, pillar="contracts",
                           reason=f"freshness delay {freshness_delay} exceeds SLA {sla_freshness}")
        # Global baseline max
        if freshness_delay > ctx.baseline["freshness_delay_max_min"]:
            return Verdict(alert=True, pillar="contracts",
                           reason=f"freshness delay {freshness_delay} exceeds baseline max")

    return Verdict(alert=False, pillar="contracts")


def check_lineage_run(payload, ctx):
    if not _budget_ok(ctx, 1.0):
        return Verdict(alert=False, pillar="lineage")

    slice_data = ctx.tools.lineage_graph_slice(payload["run_id"])
    if "error" in slice_data:
        return Verdict(alert=False, pillar="lineage")

    duration_ms = slice_data.get("duration_ms")
    if duration_ms is not None:
        if duration_ms > ctx.baseline["lineage_duration_ms_max"]:
            return Verdict(alert=True, pillar="lineage", reason="duration exceeds baseline")

    # Orphan output
    actual_downstream_count = slice_data.get("actual_downstream_count")
    if actual_downstream_count is not None:
        if actual_downstream_count == 0:
            return Verdict(alert=True, pillar="lineage", reason="orphan output")

    # Missing upstream for known jobs
    actual_upstream = slice_data.get("actual_upstream", [])
    job = payload.get("job")
    if job == "dbt:stg_orders":
        if "raw.orders" not in actual_upstream or "raw.customers" not in actual_upstream:
            return Verdict(alert=True, pillar="lineage",
                           reason="missing expected upstream for dbt:stg_orders")

    return Verdict(alert=False, pillar="lineage")


def check_feature_materialization(payload, ctx):
    if not _budget_ok(ctx, 2.0):
        return Verdict(alert=False, pillar="ai_infra")

    drift = ctx.tools.feature_drift(payload["feature_view"], payload["batch_id"])
    if "error" in drift:
        return Verdict(alert=False, pillar="ai_infra")

    mean_shift_sigma = drift.get("mean_shift_sigma")
    if mean_shift_sigma is not None:
        # Calibrated static threshold: >= 0.75 sigma perfectly separates faulty
        # from clean (FPR=0, TPR=1 on practice).  The baseline's
        # feature_mean_shift_sigma_max (0.41) is too tight — clean batches
        # naturally reach 0.4–0.7 sigma, causing many FPs.
        if mean_shift_sigma >= 0.75:
            return Verdict(alert=True, pillar="ai_infra",
                           reason=f"feature mean_shift_sigma {mean_shift_sigma} >= 0.75")

    return Verdict(alert=False, pillar="ai_infra")


def check_embedding_batch(payload, ctx):
    if not _budget_ok(ctx, 2.0):
        return Verdict(alert=False, pillar="ai_infra")

    drift = ctx.tools.embedding_drift(payload["corpus"], payload["chunk_batch_id"])
    if "error" in drift:
        return Verdict(alert=False, pillar="ai_infra")

    centroid_shift = drift.get("centroid_shift")
    if centroid_shift is not None:
        if centroid_shift > ctx.baseline["embedding_centroid_shift_max"]:
            return Verdict(alert=True, pillar="ai_infra",
                           reason=f"centroid_shift {centroid_shift} exceeds baseline")

    avg_doc_age_days = drift.get("avg_doc_age_days")
    if avg_doc_age_days is not None:
        if avg_doc_age_days > ctx.baseline["corpus_avg_doc_age_days_max"]:
            return Verdict(alert=True, pillar="ai_infra",
                           reason=f"avg_doc_age {avg_doc_age_days} exceeds baseline")

    return Verdict(alert=False, pillar="ai_infra")
