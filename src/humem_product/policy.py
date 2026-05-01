from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrievalProfile:
    """Named retrieval and learning behavior for a HuMem memory store."""

    name: str = "balanced"
    memory_weight: float = 0.65
    embedding_weight: float = 0.35
    reinforce_on_read: bool = True
    read_reinforcement: float = 0.08
    feedback_positive_amount: float = 0.36
    feedback_negative_amount: float = 0.32
    feedback_demote_layers: int = 1
    description: str = "Balanced layered memory plus optional semantic recall."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


RETRIEVAL_PROFILES: dict[str, RetrievalProfile] = {
    "balanced": RetrievalProfile(),
    "conservative": RetrievalProfile(
        name="conservative",
        memory_weight=0.82,
        embedding_weight=0.18,
        read_reinforcement=0.07,
        feedback_positive_amount=0.34,
        feedback_negative_amount=0.36,
        description="Prefer high-confidence layered memory anchors over semantic expansion.",
    ),
    "semantic": RetrievalProfile(
        name="semantic",
        memory_weight=0.45,
        embedding_weight=0.55,
        read_reinforcement=0.06,
        feedback_positive_amount=0.32,
        feedback_negative_amount=0.28,
        description="Give embedding similarity more influence while keeping layer accessibility.",
    ),
    "exploratory": RetrievalProfile(
        name="exploratory",
        memory_weight=0.55,
        embedding_weight=0.45,
        read_reinforcement=0.04,
        feedback_positive_amount=0.28,
        feedback_negative_amount=0.24,
        description="Explore broader associations with gentler read-time reinforcement.",
    ),
    "archival": RetrievalProfile(
        name="archival",
        memory_weight=0.70,
        embedding_weight=0.30,
        reinforce_on_read=False,
        read_reinforcement=0.0,
        feedback_positive_amount=0.30,
        feedback_negative_amount=0.30,
        description="Read without mutating memory state; useful for evaluation and audits.",
    ),
}


def make_retrieval_profile(
    profile: str | RetrievalProfile | dict[str, Any] | None = None,
    *,
    memory_weight: float | None = None,
    embedding_weight: float | None = None,
) -> RetrievalProfile:
    if profile is None:
        resolved = RETRIEVAL_PROFILES["balanced"]
    elif isinstance(profile, RetrievalProfile):
        resolved = profile
    elif isinstance(profile, str):
        try:
            resolved = RETRIEVAL_PROFILES[profile]
        except KeyError as exc:
            supported = ", ".join(sorted(RETRIEVAL_PROFILES))
            raise ValueError(f"unsupported retrieval profile: {profile}. Supported: {supported}") from exc
    elif isinstance(profile, dict):
        base_name = str(profile.get("name", "balanced"))
        base = RETRIEVAL_PROFILES.get(base_name, RETRIEVAL_PROFILES["balanced"])
        allowed = set(RetrievalProfile.__dataclass_fields__)
        resolved = replace(base, **{key: value for key, value in profile.items() if key in allowed})
    else:
        raise TypeError("retrieval profile must be a name, mapping, RetrievalProfile, or None")

    overrides: dict[str, Any] = {}
    if memory_weight is not None:
        overrides["memory_weight"] = memory_weight
    if embedding_weight is not None:
        overrides["embedding_weight"] = embedding_weight
    if overrides:
        resolved = replace(resolved, **overrides)

    _validate_profile(resolved)
    return resolved


def _validate_profile(profile: RetrievalProfile) -> None:
    if profile.memory_weight < 0 or profile.embedding_weight < 0:
        raise ValueError("retrieval weights must be non-negative")
    if profile.memory_weight == 0 and profile.embedding_weight == 0:
        raise ValueError("at least one retrieval weight must be positive")
    if profile.read_reinforcement < 0:
        raise ValueError("read_reinforcement must be non-negative")
    if profile.feedback_positive_amount < 0 or profile.feedback_negative_amount < 0:
        raise ValueError("feedback amounts must be non-negative")
    if profile.feedback_demote_layers < 0:
        raise ValueError("feedback_demote_layers must be non-negative")
