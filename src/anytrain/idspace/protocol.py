from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class EmbeddingProtocol(Protocol):
    @property
    def num_embeddings(self) -> int: ...

    @property
    def embedding_dim(self) -> int: ...

    @property
    def weight(self) -> Tensor: ...
