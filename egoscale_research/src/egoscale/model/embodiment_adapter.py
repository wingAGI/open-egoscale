from __future__ import annotations

from typing import Callable, Dict, Iterable, List

import torch


class EmbodimentAdapter:
    def __init__(self, embodiments: Iterable[str]) -> None:
        self.embodiments = list(embodiments)
        self.index = {name: idx for idx, name in enumerate(self.embodiments)}

    def unique(self, embodiment_ids: List[str]) -> List[str]:
        return sorted(set(embodiment_ids), key=self.index.get)

    def gather_indices(self, embodiment_ids: List[str], embodiment_name: str) -> torch.Tensor:
        indices = [idx for idx, current in enumerate(embodiment_ids) if current == embodiment_name]
        return torch.tensor(indices, dtype=torch.long)

    def route_batch(self, embodiment_ids: List[str], fn: Callable[[str, torch.Tensor], None]) -> None:
        for embodiment_name in self.unique(embodiment_ids):
            indices = self.gather_indices(embodiment_ids, embodiment_name)
            fn(embodiment_name, indices)
