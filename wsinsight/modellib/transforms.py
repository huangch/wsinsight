"""PyTorch image classification transform."""

from __future__ import annotations

from torchvision import transforms
from wsinfer_zoo.client import TransformConfigurationItem

class Scale:
    def __init__(self, upper: float=1.0, lower: float = 0) -> None:
        self.upper = upper
        self.lower = lower

    def __call__(self, input):
        input -= input.min()
        input /= input.max()
        input *= (self.upper - self.lower)
        input += self.lower
        return input
    
# The subset of transforms known to the wsinsight config spec.
# This can be expanded in the future as needs arise.
_name_to_tv_cls = {
    "Resize": transforms.Resize,
    "ToTensor": transforms.ToTensor,
    "Normalize": transforms.Normalize,
    "Scale": Scale,
}

def make_compose_from_transform_config(
    list_of_transforms: list[TransformConfigurationItem],
) -> transforms.Compose:
    """Create a torchvision Compose instance from configuration of transforms."""
    all_t: list = []
    for t in list_of_transforms:
        cls = _name_to_tv_cls[t.name]
        kwargs = t.arguments or {}
        all_t.append(cls(**kwargs))
    return transforms.Compose(all_t)
