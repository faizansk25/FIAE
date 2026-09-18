"""Proposal source 4: experience-guided priors (doc 04 + doc 06).

Retrieves similar historical cases from the ExperienceStore and turns the
retrieved feature-family priors into concrete FeatureProposals: for each
operator family with a posterior success rate above policy, the best-matching
columns of that family are proposed with the family's operators.

Priors are *hints*, not verdicts (doc 04): they only add candidates to the
pool; the funnel F0-F6 still has to earn every acceptance. With an empty
store or low-confidence retrieval this source contributes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..contracts import DatasetProfile, SemanticType, Task
from ..experience import (
    DatasetMetaFeatures,
    ExperienceStore,
    extract_meta_features,
    retrieve_priors,
)
from ..experience.retrieval import RetrievalQuery
from ..search.triggers import FeatureProposal, _raw_id

# semantic types that map to each operator family
_FAMILY_SEMANTICS: dict[str, tuple] = {
    "numeric": (
        SemanticType.CONTINUOUS_NUMERIC, SemanticType.COUNT,
        SemanticType.PERCENTAGE, SemanticType.CURRENCY,
    ),
    "datetime": (SemanticType.DATETIME,),
    "temporal": (
        SemanticType.CONTINUOUS_NUMERIC, SemanticType.COUNT,
        SemanticType.PERCENTAGE, SemanticType.CURRENCY,
    ),
}

# canonical representative operator per family (kept cheap for probing)
_FAMILY_OPS: dict[str, tuple] = {
    "numeric": ("log1p", "square"),
    "numeric interaction": ("safe_ratio", "difference"),
    "datetime": ("day_of_week", "month"),
    "temporal": ("lag", "rolling_mean"),
}


@dataclass
class PriorPolicy:
    """Source-4 policy thresholds (policy, not universal constants)."""

    min_posterior: float = 0.5      # family success rate needed to propose
    min_confidence: float = 0.1     # minimum retrieval confidence
    max_columns_per_family: int = 3
    max_proposals: int = 16


def propose_from_experience(
    profile: DatasetProfile,
    store: ExperienceStore,
    task: Task,
    policy: Optional[PriorPolicy] = None,
) -> list[FeatureProposal]:
    """Turn retrieved feature-family priors into concrete proposals."""
    policy = policy or PriorPolicy()
    meta: DatasetMetaFeatures = extract_meta_features(profile)
    query = RetrievalQuery(task=task, meta_features=meta)
    prior = retrieve_priors(store, query)
    if prior.confidence < policy.min_confidence or prior.source_count == 0:
        return []

    out: list[FeatureProposal] = []
    for fam, fam_prior in prior.feature_family_priors.items():
        fam = fam_prior.family or fam
        ops = _FAMILY_OPS.get(fam)
        semantics = _FAMILY_SEMANTICS.get(fam)
        if not ops or not semantics:
            continue
        if fam_prior.posterior_mean < policy.min_posterior:
            continue
        cols = [
            c for c in profile.columns
            if c.semantic_type in semantics and c.null_fraction < 0.5
        ]
        for col in cols[: policy.max_columns_per_family]:
            for op in ops:
                if len(out) >= policy.max_proposals:
                    return out
                out.append(
                    FeatureProposal(
                        op=op, inputs=[_raw_id(col.name)],
                        params={},
                        source="experience:prior",
                        trigger_reason=(
                            f"family '{fam}' posterior "
                            f"{fam_prior.posterior_mean:.2f} "
                            f"({fam_prior.successes}/{fam_prior.attempts})"
                        ),
                        depth=1,
                    )
                )
    return out
