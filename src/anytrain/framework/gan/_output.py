from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor

from anytrain._compat import strict_zip

Features = list[list[Tensor]]


def split(output: object) -> tuple[Features, list[Tensor]]:
    if isinstance(output, Tensor):
        raise TypeError(
            "discriminator must return a sequence of branch sequences; "
            "for a simple discriminator return [[logits]], not logits."
        )
    if not isinstance(output, Sequence) or isinstance(output, (str, bytes)):
        raise TypeError("discriminator output must be a sequence of branch sequences.")
    if not output:
        raise ValueError("discriminator output must contain at least one branch.")

    features: Features = []
    logits: list[Tensor] = []
    for branch_index, raw_branch in enumerate(output):
        branch = _branch(raw_branch, index=branch_index)
        features.append(branch[:-1])
        logits.append(branch[-1])
    return features, logits


def validate_matching_features(
    fake: Features,
    real: Features,
    *,
    require_features: bool,
) -> None:
    if len(fake) != len(real):
        raise ValueError("fake and real discriminator outputs must have the same branch count.")

    found_feature = False
    for branch_index, (fake_branch, real_branch) in enumerate(strict_zip(fake, real)):
        if len(fake_branch) != len(real_branch):
            raise ValueError(
                "fake and real discriminator outputs must have the same feature count "
                f"in branch {branch_index}."
            )
        for feature_index, (fake_feature, real_feature) in enumerate(
            strict_zip(fake_branch, real_branch)
        ):
            found_feature = True
            if fake_feature.shape != real_feature.shape:
                raise ValueError(
                    "fake and real feature maps must have the same shape: "
                    f"branch {branch_index}, feature {feature_index}."
                )

    if require_features and not found_feature:
        raise ValueError("discriminator outputs do not contain feature maps.")


def _branch(value: object, *, index: int) -> list[Tensor]:
    if isinstance(value, Tensor):
        raise TypeError(
            f"discriminator branch {index} must be a sequence of tensors; "
            "for branch logits return [logits], so a simple discriminator returns [[logits]]."
        )
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"discriminator branch {index} must be a sequence of tensors.")
    if not value:
        raise ValueError(f"discriminator branch {index} must contain at least logits.")

    tensors: list[Tensor] = []
    for tensor_index, tensor in enumerate(value):
        if not isinstance(tensor, Tensor):
            raise TypeError(
                f"discriminator branch {index} item {tensor_index} must be a tensor."
            )
        tensors.append(tensor)
    return tensors
