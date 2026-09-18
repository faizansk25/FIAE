"""Experience Store, Meta-Learning, and Retrieval (doc 06).

The Experience Store keeps context, actions, outcomes, cost, and failure
reasons from prior runs. Historical knowledge changes search order only;
current-dataset experiments remain authoritative.
"""

from .case import (
    CaseRecord,
    CaseResult,
    CaseContext,
    CaseAction,
    CaseCost,
    CaseDiagnosis,
    FailureTag,
)
from .metafeatures import (
    MetaFeatureFamily,
    DatasetMetaFeatures,
    extract_meta_features,
)
from .store import ExperienceStore, StoreConfig
from .priors import (
    RetrievalPrior,
    FeatureFamilyPrior,
    ModelFamilyPrior,
    bayesian_posterior,
)
from .retrieval import (
    retrieve_priors,
    R3Ranker,
    bayesian_smooth,
    RetrievalEngine,
    RetrievalQuery,
)

__all__ = [
    # Case schema
    "CaseRecord",
    "CaseResult",
    "CaseContext",
    "CaseAction",
    "CaseCost",
    "CaseDiagnosis",
    "FailureTag",
    # Meta-features
    "MetaFeatureFamily",
    "DatasetMetaFeatures",
    "extract_meta_features",
    # Store
    "ExperienceStore",
    "StoreConfig",
    # Priors
    "RetrievalPrior",
    "FeatureFamilyPrior",
    "ModelFamilyPrior",
    "bayesian_posterior",
    # Retrieval
    "retrieve_priors",
    "RetrievalEngine",
    "RetrievalQuery",
    "R3Ranker",
    "bayesian_smooth",
]
