"""Model registry helpers plus TorchScript loading utilities."""

from __future__ import annotations

import os
from pathlib import Path
import dataclasses
import warnings
from typing import Callable

import torch
import wsinfer_zoo
from wsinfer_zoo.client import HFModelTorchScript
from wsinfer_zoo.client import Model


@dataclasses.dataclass
class LocalModelTorchScript(Model):
    """Typed stand-in for locally stored TorchScript bundles."""

    ...


def get_registered_model(name: str) -> HFModelTorchScript:
    """Resolve a model name to the corresponding TorchScript handle."""

    # registry = wsinfer_zoo.client.load_registry(registry_file=Path.home() / ".wsinfer-zoo" if not os.getenv("WSINFER_ZOO_DIR", default=None) else Path(os.getenv("WSINFER_ZOO_DIR", default=None)) / "wsinfer-zoo-registry.json")
    registry = wsinfer_zoo.client.load_registry(
        registry_file=Path(os.getenv("WSINFER_ZOO_REGISTRY_PATH", default=None)) \
            if Path(os.getenv("WSINFER_ZOO_REGISTRY_PATH", default=None)).exists() \
            else Path(wsinfer_zoo.client.WSINFER_ZOO_REGISTRY_DEFAULT_PATH) \
            if Path(wsinfer_zoo.client.WSINFER_ZOO_REGISTRY_DEFAULT_PATH).exists() \
            else None)
    
    model = registry.get_model_by_name(name=name)
    return model.load_model_torchscript()

# class ScriptWrapper(torch.nn.Module):
#     def __init__(self, path):
#         super().__init__()
#         self.path = path
#         self.models = {}
#     def forward(self, x):
#         dev = x.device
#         if dev not in self.models:
#             self.models[dev] = torch.jit.load(self.path, map_location=dev).eval()
        # return self.models[dev](x)



class TSPerDevice(torch.nn.Module):
    """Lazy TorchScript loader that preserves constants per target device."""

    def __init__(self, ts_path: str, #, device: torch.Device
                 ):
        super().__init__()
        self.ts_path = ts_path
        self._per_device = {}  # device_str -> ScriptModule

    def _get(self, dev):
        k = str(dev)
        if k not in self._per_device:
            # CRUCIAL: load with map_location so CONSTANTS land on this GPU
            m = torch.jit.load(self.ts_path, map_location=dev).eval()
            self._per_device[k] = m
        return self._per_device[k]

    def forward(self, *args, **kwargs):
        """Dispatch the scripted model on whichever device the inputs reside."""

        # infer device from first tensor-like input
        def pick_device(obj):
            if torch.is_tensor(obj): return obj.device
            if isinstance(obj, (list, tuple)):
                for z in obj:
                    d = pick_device(z)
                    if d is not None: return d
            if isinstance(obj, dict):
                for z in obj.values():
                    d = pick_device(z)
                    if d is not None: return d
            return None
        dev = pick_device(args[0] if args else next(iter(kwargs.values())))
        
        torch.cuda.set_device(dev.index or 0)
        torch.set_default_device(f"cuda:{dev.index or 0}")
        
        
        m = self._get(dev)
        return m(*args, **kwargs)


def get_pretrained_torch_module(
    model: HFModelTorchScript | LocalModelTorchScript,
    # device: torch.device = "cpu",
) -> torch.nn.Module:
    """Get a PyTorch Module with weights loaded."""
    # mod: torch.nn.Module = torch.jit.load(model.model_path, map_location="cpu")
    # mod: torch.nn.Module = torch.jit.load(model.model_path, map_location=device)
    
    mod: torch.nn.Module = TSPerDevice(model.model_path)
    
    if not isinstance(mod, torch.nn.Module):
        raise TypeError(
            "expected the loaded object to be a subclass of torch.nn.Module but got"
            f" {type(mod)}."
        )
    return mod


def jit_compile(
    model: torch.nn.Module,
) -> torch.jit.ScriptModule | torch.nn.Module | Callable:
    """JIT-compile a model for inference.

    A torchscript model may be JIT compiled here as well.
    """
    noncompiled = model
    device = next(model.parameters()).device
    # Attempt to script. If it fails, return the original.
    test_input = torch.ones(1, 3, 224, 224).to(device)
    w = "Warning: could not JIT compile the model. Using non-compiled model instead."

    # PyTorch 2.x has torch.compile but it does not work when applied
    # to TorchScript models.
    if hasattr(torch, "compile") and not isinstance(model, torch.jit.ScriptModule):
        # Try to get the most optimized model.
        try:
            return torch.compile(model, fullgraph=True, mode="max-autotune")
        except Exception:
            pass
        try:
            return torch.compile(model, mode="max-autotune")
        except Exception:
            pass
        try:
            return torch.compile(model)
        except Exception:
            warnings.warn(w, stacklevel=1)
            return noncompiled
    # For pytorch 1.x, use torch.jit.script.
    else:
        try:
            mjit = torch.jit.script(model)
            with torch.no_grad():
                mjit(test_input)
        except Exception:
            warnings.warn(w, stacklevel=1)
            return noncompiled
        # Now that we have scripted the model, try to optimize it further. If that
        # fails, return the scripted model.
        try:
            mjit_frozen = torch.jit.freeze(mjit)
            mjit_opt = torch.jit.optimize_for_inference(mjit_frozen)
            with torch.no_grad():
                mjit_opt(test_input)
            return mjit_opt
        except Exception:
            return mjit  # type: ignore
