"""Dataset utilities for loading whole-slide patch tensors on demand."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
from typing import Sequence

import h5py
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
import histomicstk as htk

from wsinsight.wsi import get_wsi_cls
from ..uri_path import URIPath

EPSILON = 1e-8
I_0 = 255

def _read_patch_coords(path: str | Path) -> npt.NDArray[np.int_]:
    """Read HDF5 file of patch coordinates are return numpy array.

    Returned array has shape (num_patches, 4). Each row has values
    [minx, miny, width, height].
    """
    with h5py.File(path, mode="r") as f:
        coords: npt.NDArray[np.int_] = f["/coords"][()]
        coords_metadata = f["/coords"].attrs
        if "patch_level" not in coords_metadata.keys():
            raise KeyError(
                "Could not find required key 'patch_level' in hdf5 of patch "
                "coordinates. Has the version of CLAM been updated?"
            )
        patch_level = coords_metadata["patch_level"]
        if patch_level != 0:
            raise NotImplementedError(
                f"This script is designed for patch_level=0 but got {patch_level}"
            )
        if coords.ndim != 2:
            raise ValueError(f"expected coords to have 2 dimensions, got {coords.ndim}")
        if coords.shape[1] != 2:
            raise ValueError(
                f"expected second dim of coords to have len 2 but got {coords.shape[1]}"
            )

        if "patch_size" not in coords_metadata.keys():
            raise KeyError("expected key 'patch_size' in attrs of coords dataset")
        # Append width and height values to the coords, so now each row is
        # [minx, miny, width, height]
        wh = np.full_like(coords, coords_metadata["patch_size"])
        coords = np.concatenate((coords, wh), axis=1)
        tile_dim = coords_metadata["tile_dim"] if "tile_dim" in coords_metadata.keys() else None
        # step_size = coords_metadata["step_size"]
        patch_size = coords_metadata["patch_size"]
        # halo_size = coords_metadata["halo_size"]
        
        # object_based = coords_metadata["object_based"]
        # object_end2end = coords_metadata["object_end2end"]
        
    # return coords, tile_dim, step_size, patch_size, halo_size, # object_based, object_end2end
    return coords, tile_dim, patch_size, # halo_size, # object_based, object_end2end


# class WholeSlideImagePatches(torch.utils.data.Dataset):
#     """Dataset of one whole slide image.
#
#     This object retrieves patches from a whole slide image on the fly.
#
#     Parameters
#     ----------
#     wsi_path : str, Path
#         Path to whole slide image file.
#     patch_path : str, Path
#         Path to npy file with coordinates of input image.
#     transform : callable, optional
#         A callable to modify a retrieved patch. The callable must accept a
#         PIL.Image.Image instance and return a torch.Tensor.
#     """
#
#     def __init__(
#         self,
#         wsi_path: str | Path,
#         patch_path: str | Path,
#         transform: Callable[[Image.Image], torch.Tensor] | None = None,
#         W_est: np.Array | None = None,
#         W_def: np.Array | None = None,
#     ):
#         self.wsi_path = wsi_path
#         self.patch_path = patch_path
#         self.transform = transform
#
#         # self.color_mode = color_mode
#         self.W_est = W_est
#         self.W_def = W_def
#
#         assert Path(wsi_path).exists(), "wsi path not found"
#         assert Path(patch_path).exists(), "patch path not found"
#
#         # (self.patches, self.tile_dim, self.step_size, self.patch_size, self.halo_size,) = _read_patch_coords(self.patch_path)
#         # (self.patches, self.tile_dim, self.patch_size, self.halo_size,) = _read_patch_coords(self.patch_path)
#         (self.patches, self.tile_dim, self.patch_size,) = _read_patch_coords(self.patch_path)
#         if self.patches.size == 0:
#             raise ValueError(f"No patches were found in {self.patch_path}")
#         assert self.patches.ndim == 2, "expected 2D array of patch coordinates"
#         # x, y, width, height
#         assert self.patches.shape[1] == 4, "expected second dimension to have len 4"
#
#     def worker_init(self, worker_id: int | None = None) -> None:
#         del worker_id
#         wsi_reader = get_wsi_cls()
#         self.slide = wsi_reader(self.wsi_path)
#
#     def __len__(self) -> int:
#         return self.patches.shape[0]
#
#     def __getitem__(self, idx: int) -> tuple[Image.Image | torch.Tensor, torch.Tensor]:
#         coords: Sequence[int] = self.patches[idx]
#         assert len(coords) == 4, "expected 4 coords (minx, miny, width, height)"
#         minx, miny, width, height = coords
#
#         patch_im = self.slide.read_region(
#             # 09/05/2025: huangch, the weidth and height have to be integer. Here, because of coords, they are assigned as np.float32
#             # location=(minx, miny), level=0, size=(width, height)
#             # location=(int(minx-self.halo_size), int(miny-self.halo_size)), level=0, size=(int(width+2*self.halo_size), int(height+2*self.halo_size))
#             location=(int(minx), int(miny)), level=0, size=(int(width), int(height))
#
#         )
#
#         patch_im = patch_im.convert("RGB")        
#
#         if self.W_est is not None and self.W_def is not None: 
#             patch_im = np.array(patch_im).astype(np.float32)
#             patch_im = htk.preprocessing.color_normalization.deconvolution_based_normalization(patch_im+EPSILON, W_source=self.W_est, W_target=self.W_def)
#             patch_im = Image.fromarray(patch_im, mode='RGB')
#
#         if self.transform is not None:
#             patch_im = self.transform(patch_im)
#         else:
#             patch_im = np.transpose(np.array(patch_im), (2, 0, 1))
#
#         # if self.color_mode == "BGR":
#         #     patch_im = patch_im[[2,1,0],:,:] 
#
#         return patch_im, torch.as_tensor([minx, miny, width, height])


class WholeSlideImagePatches(torch.utils.data.Dataset):
    """Dataset of one whole slide image.

    This object retrieves patches from a whole slide image on the fly.

    Parameters
    ----------
    wsi_path : str, Path
        Path to whole slide image file.
    patch_path : str, Path
        Path to HDF5 file with coordinates (and optionally images).
    transform : callable, optional
        A callable to modify a retrieved patch. The callable must accept a
        PIL.Image.Image instance and return a torch.Tensor.
    """

    def __init__(
        self,
        wsi_path: URIPath | None,
        patch_path: URIPath,
        use_hdf5_images: bool,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        W_est: np.ndarray | None = None,
        W_def: np.ndarray | None = None,
    ):
        self.wsi_path = wsi_path
        self.patch_path = patch_path
        self.use_hdf5_images = use_hdf5_images
        self.transform = transform

        self.W_est = W_est
        self.W_def = W_def

        assert self.use_hdf5_images or self.wsi_path.exists(), "wsi path not found"
        assert self.patch_path.exists(), "patch path not found"

        # coords: (N, 4) = [minx, miny, width, height]
        (self.patches, self.tile_dim, self.patch_size,) = _read_patch_coords(self.patch_path)
        if self.patches.size == 0:
            raise ValueError(f"No patches were found in {self.patch_path}")
        assert self.patches.ndim == 2, "expected 2D array of patch coordinates"
        assert self.patches.shape[1] == 4, "expected second dimension to have len 4"

        # Will be initialized in worker_init
        self.slide = None
        self._h5_file: h5py.File | None = None
        self._images_dset = None
        

    def worker_init(self, worker_id: int | None = None) -> None:
        """Open slide and HDF5 handles in each worker process."""

        # avoid warning about unused
        del worker_id

        # Open WSI reader in this worker
        wsi_reader = get_wsi_cls() if not self.use_hdf5_images else None
        self.slide = wsi_reader(self.wsi_path) if not self.use_hdf5_images else None

        # Open HDF5 in this worker (separate handle per process)
        # and check if /images exists and is compatible.
        self.use_hdf5_images = False
        self._images_dset = None

        # It’s okay to leave the file open for the life of the worker
        self._h5_file = h5py.File(self.patch_path, mode="r")

        if "/images" in self._h5_file:
            imgs = self._h5_file["/images"]
            # Basic sanity checks
            if imgs.ndim != 4:
                # Expected (N, H, W, C) or (N, C, H, W)
                print(f"[WholeSlideImagePatches] /images has unexpected ndim={imgs.ndim}, ignoring.")
            elif imgs.shape[0] != self.patches.shape[0]:
                print(
                    f"[WholeSlideImagePatches] /images length ({imgs.shape[0]}) "
                    f"!= coords length ({self.patches.shape[0]}), ignoring."
                )
            else:
                self._images_dset = imgs
                self.use_hdf5_images = True
                # print(
                #     f"[WholeSlideImagePatches] Using /images from HDF5 "
                #     f"({imgs.shape[0]} patches)."
                # )
        else:
            # No images in HDF5 → fall back to slide.read_region
            self.use_hdf5_images = False

    def __len__(self) -> int:
        """Return the number of available patch coordinates."""

        return self.patches.shape[0]

    def _get_patch_from_hdf5(self, idx: int) -> Image.Image:
        """Load a patch from /images dataset and convert to PIL.Image."""
        assert self._images_dset is not None
        arr = self._images_dset[idx]  # should be np.ndarray

        # Expect either (H, W, 3) or (3, H, W)
        if arr.ndim != 3:
            raise ValueError(f"/images[idx] must be 3D (H,W,3 or C,H,W), got shape {arr.shape}")

        if arr.shape[-1] == 3:
            # (H, W, 3)
            img_arr = arr
        elif arr.shape[0] == 3 and arr.shape[-1] != 3:
            # (3, H, W) → (H, W, 3)
            img_arr = np.transpose(arr, (1, 2, 0))
        else:
            raise ValueError(
                f"/images[idx] has unsupported shape {arr.shape}. "
                "Expected (H, W, 3) or (3, H, W)."
            )

        if img_arr.dtype != np.uint8:
            # You *can* relax this to auto-scale, but best is to save as uint8.
            img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)

        return Image.fromarray(img_arr, mode="RGB")

    def _get_patch_from_wsi(self, idx: int) -> Image.Image:
        """Read patch from WSI using coordinates."""
        coords: Sequence[int] = self.patches[idx]
        assert len(coords) == 4, "expected 4 coords (minx, miny, width, height)"
        minx, miny, width, height = coords

        patch_im = self.slide.read_region(
            location=(int(minx), int(miny)),
            level=0,
            size=(int(width), int(height)),
        )
        return patch_im.convert("RGB")

    def __getitem__(self, idx: int) -> tuple[Image.Image | torch.Tensor, torch.Tensor]:
        """Load one patch tensor plus its corresponding slide coordinates."""

        # 1) Get patch image (HDF5 first; else WSI)
        if self.use_hdf5_images and self._images_dset is not None:
            patch_im = self._get_patch_from_hdf5(idx)
        else:
            patch_im = self._get_patch_from_wsi(idx)

        # 2) Optional stain normalization
        if self.W_est is not None and self.W_def is not None:
            patch_arr = np.array(patch_im).astype(np.float32)
            patch_arr = htk.preprocessing.color_normalization.deconvolution_based_normalization(
                patch_arr + EPSILON,
                W_source=self.W_est,
                W_target=self.W_def,
            )
            patch_im = Image.fromarray(patch_arr.astype(np.uint8), mode="RGB")

        # 3) Transform → tensor
        if self.transform is not None:
            patch_tensor = self.transform(patch_im)
        else:
            # fallback: CHW float tensor
            patch_arr = np.transpose(np.array(patch_im), (2, 0, 1))  # (C,H,W)
            patch_tensor = torch.from_numpy(patch_arr)

        # 4) Return coords as before
        minx, miny, width, height = self.patches[idx]
        coord_tensor = torch.as_tensor([minx, miny, width, height])

        return patch_tensor, coord_tensor

    def __del__(self):
        # Best-effort cleanup of HDF5 file handle
        try:
            if hasattr(self, "_h5_file") and self._h5_file is not None:
                self._h5_file.close()
        except Exception:
            pass

