"""
Patchwork image registration (translation / affine / homography).

Implements the core idea from:
    Hattori & Komiyama, "PatchWarp: Corrections of non-uniform image
    distortions in two-photon calcium imaging data by patchwork affine
    transformations", Cell Reports Methods, 2022.
    https://doi.org/10.1016/j.crmeth.2022.100205

The approach:
  1. Split the reference (template) and target images into a grid of
     overlapping patches.
  2. For each patch, fit a per-patch transform that maximizes the
     Enhanced Correlation Coefficient (ECC) between the template patch
     and the target patch (cv2.findTransformECC). The transform model
     is selectable: translation, affine (default, as in the paper), or
     homography (projective).
  3. Warp each target patch with its own patch-specific transform.
  4. Stitch the warped patches back together with linear (feathered)
     blending in the overlap regions to avoid seams.

This lets a single global transform (which can't model local /
nonuniform distortion) be replaced by a smoothly-varying field of local
transforms -- i.e. a piecewise approximation of a nonrigid warp. The
paper uses affine patches; translation is a cheaper/more-constrained
option for patches with little local rotation/scale/shear, and
homography is a richer option when patches show perspective-like
(keystone) distortion rather than pure affine distortion.

Dependencies: numpy, opencv-python
"""

from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional


#: Map user-facing motion-model names to OpenCV ECC motion types, the
#: identity-matrix shape each uses, and whether warping it needs
#: warpAffine (2x3) or warpPerspective (3x3).
_MOTION_MODELS = {
    "translation": {"cv2_type": cv2.MOTION_TRANSLATION, "shape": (2, 3), "perspective": False},
    "affine": {"cv2_type": cv2.MOTION_AFFINE, "shape": (2, 3), "perspective": False},
    "homography": {"cv2_type": cv2.MOTION_HOMOGRAPHY, "shape": (3, 3), "perspective": True},
}


@dataclass
class PatchResult:
    """Holds the fitted transform and bookkeeping for one patch."""
    row: int
    col: int
    y0: int
    y1: int
    x0: int
    x1: int
    warp_matrix: np.ndarray          # 2x3 (translation/affine) or 3x3 (homography)
    converged: bool
    cc: float                        # final ECC correlation coefficient


class PatchworkAffineRegistration:
    """
    Estimate and apply a grid of local transforms to register a target
    image to a reference (template) image, correcting for spatially
    nonuniform distortion.

    Parameters
    ----------
    patch_size : int or (int, int)
        Size (height, width) of each patch in pixels. Smaller patches
        capture more local distortion but are noisier to fit; this
        should be set based on how fast/local the distortion is.
    overlap : float
        Fractional overlap between neighboring patches (0-0.9). Overlap
        is required for seamless feathered blending; ~0.5 is a
        reasonable default.
    motion_type : {"translation", "affine", "homography"}
        Per-patch transform model fit by ECC.
          - "translation": 2 DOF (x, y shift only). Cheapest, most
            robust on small/low-contrast/noisy patches, but can't
            correct any rotation/scale/shear/perspective within a patch.
          - "affine" (default): 6 DOF (rotation, scale, shear,
            translation). Matches the PatchWarp paper.
          - "homography": 8 DOF projective transform. Strictly more
            expressive than affine (captures keystone/perspective-like
            distortion within a patch) but needs more well-conditioned
            structure in the patch to fit reliably -- prefer larger
            patches and/or higher n_iterations if you see many patches
            failing to converge.
    n_iterations : int
        Max ECC solver iterations per patch.
    eps : float
        ECC solver convergence threshold.
    gaussian_blur_sigma : float or None
        Optional Gaussian blur applied to each patch before ECC fitting
        (helps the gradient-based ECC optimizer on noisy data, similar
        to the smoothing PatchWarp applies before fitting).
    """

    def __init__(
        self,
        patch_size: Tuple[int, int] | int = 128,
        overlap: float = 0.5,
        motion_type: str = "affine",
        n_iterations: int = 200,
        eps: float = 1e-6,
        gaussian_blur_sigma: Optional[float] = 1.0,
    ):
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_h, self.patch_w = patch_size
        if not (0 <= overlap < 1):
            raise ValueError("overlap must be in [0, 1)")
        self.overlap = overlap
        if motion_type not in _MOTION_MODELS:
            raise ValueError(
                f"motion_type must be one of {list(_MOTION_MODELS)}, got {motion_type!r}"
            )
        self.motion_type = motion_type
        self._model = _MOTION_MODELS[motion_type]
        self.n_iterations = n_iterations
        self.eps = eps
        self.gaussian_blur_sigma = gaussian_blur_sigma

    # ------------------------------------------------------------------
    # Patch grid geometry
    # ------------------------------------------------------------------
    def _patch_starts(self, length: int, patch_len: int) -> List[int]:
        """Starting coordinates for patches covering `length` pixels."""
        stride = max(1, int(round(patch_len * (1 - self.overlap))))
        starts = list(range(0, max(length - patch_len, 0) + 1, stride))
        # Ensure the final patch reaches the edge of the image.
        last_start = length - patch_len
        if last_start > 0 and (not starts or starts[-1] != last_start):
            starts.append(last_start)
        if not starts:
            starts = [0]
        return starts

    def _build_grid(self, shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        h, w = shape
        patch_h = min(self.patch_h, h)
        patch_w = min(self.patch_w, w)
        y_starts = self._patch_starts(h, patch_h)
        x_starts = self._patch_starts(w, patch_w)
        boxes = []
        for y0 in y_starts:
            for x0 in x_starts:
                boxes.append((y0, y0 + patch_h, x0, x0 + patch_w))
        return boxes

    # ------------------------------------------------------------------
    # Per-patch ECC affine fit
    # ------------------------------------------------------------------
    def _fit_patch(
        self,
        template_patch: np.ndarray,
        target_patch: np.ndarray,
        init_warp: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool, float]:
        """Fit a warp_matrix (shape depends on self.motion_type) mapping target_patch -> template_patch."""
        t = template_patch.astype(np.float32)
        g = target_patch.astype(np.float32)

        if self.gaussian_blur_sigma:
            k = max(3, int(2 * round(3 * self.gaussian_blur_sigma) + 1))
            t = cv2.GaussianBlur(t, (k, k), self.gaussian_blur_sigma)
            g = cv2.GaussianBlur(g, (k, k), self.gaussian_blur_sigma)

        warp_matrix = (
            init_warp.copy()
            if init_warp is not None
            else np.eye(*self._model["shape"], dtype=np.float32)
        )
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.n_iterations,
            self.eps,
        )
        try:
            cc, warp_matrix = cv2.findTransformECC(
                t, g, warp_matrix, self._model["cv2_type"], criteria, None, 5
            )
            converged = True
        except cv2.error:
            # ECC failed to converge (e.g. low-contrast/uniform patch) --
            # fall back to identity (or the initializer) for this patch.
            cc = -1.0
            converged = False

        return warp_matrix, converged, cc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self, template: np.ndarray, target: np.ndarray
    ) -> List[PatchResult]:
        """
        Fit one affine transform per patch, aligning `target` to
        `template`. Both images must be single-channel (grayscale) and
        the same shape.
        """
        if template.shape != target.shape:
            raise ValueError("template and target must have the same shape")

        boxes = self._build_grid(template.shape)
        results: List[PatchResult] = []
        for idx, (y0, y1, x0, x1) in enumerate(boxes):
            t_patch = template[y0:y1, x0:x1]
            g_patch = target[y0:y1, x0:x1]
            warp_matrix, converged, cc = self._fit_patch(t_patch, g_patch)
            row = y0
            col = x0
            results.append(
                PatchResult(row, col, y0, y1, x0, x1, warp_matrix, converged, cc)
            )
        self._last_shape = template.shape
        return results

    def apply(
        self,
        target: np.ndarray,
        patch_results: List[PatchResult],
        fill_value: float = 0.0,
    ) -> np.ndarray:
        """
        Warp `target` (which may be a different frame than what `fit`
        was called on, e.g. another channel or another frame of a
        movie) using the previously-fitted per-patch affine transforms,
        and stitch the results with feathered blending.
        """
        h, w = target.shape
        accum = np.zeros((h, w), dtype=np.float64)
        weight = np.zeros((h, w), dtype=np.float64)

        for r in patch_results:
            ph, pw = r.y1 - r.y0, r.x1 - r.x0
            patch = target[r.y0:r.y1, r.x0:r.x1]

            if r.converged:
                if self._model["perspective"]:
                    warped = cv2.warpPerspective(
                        patch.astype(np.float32),
                        r.warp_matrix,
                        (pw, ph),
                        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=fill_value,
                    )
                else:
                    warped = cv2.warpAffine(
                        patch.astype(np.float32),
                        r.warp_matrix,
                        (pw, ph),
                        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=fill_value,
                    )
            else:
                # Fitting failed for this patch (e.g. blank region) --
                # pass it through unchanged rather than corrupting it.
                warped = patch.astype(np.float32)

            w_mask = self._feather_weight(ph, pw)
            accum[r.y0:r.y1, r.x0:r.x1] += warped * w_mask
            weight[r.y0:r.y1, r.x0:r.x1] += w_mask

        weight[weight == 0] = 1.0
        stitched = accum / weight
        return stitched.astype(target.dtype)

    def register(
        self, template: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, List[PatchResult]]:
        """Convenience: fit + apply in one call. Returns (registered_image, patch_results)."""
        patch_results = self.fit(template, target)
        registered = self.apply(target, patch_results)
        return registered, patch_results

    # ------------------------------------------------------------------
    # Blending weights
    # ------------------------------------------------------------------
    @staticmethod
    def _feather_weight(ph: int, pw: int) -> np.ndarray:
        """
        Separable ramp (triangular / Hann-like) weighting that tapers to
        zero at patch edges, so overlapping patches blend smoothly
        instead of showing seams.
        """
        wy = np.hanning(ph) if ph > 1 else np.ones(1)
        wx = np.hanning(pw) if pw > 1 else np.ones(1)
        wy = np.clip(wy, 1e-3, None)  # avoid exact-zero weight at edges
        wx = np.clip(wx, 1e-3, None)
        return np.outer(wy, wx)


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Patchwork affine registration of a target image to a template image."
    )
    parser.add_argument("template", help="Path to the reference/template image")
    parser.add_argument("target", help="Path to the image to be registered")
    parser.add_argument("output", help="Path to save the registered image")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument(
        "--motion-type",
        choices=["translation", "affine", "homography"],
        default="affine",
    )
    args = parser.parse_args()

    template_img = cv2.imread(args.template, cv2.IMREAD_GRAYSCALE)
    target_img = cv2.imread(args.target, cv2.IMREAD_GRAYSCALE)
    if template_img is None or target_img is None:
        raise FileNotFoundError("Could not read one of the input images")

    registrar = PatchworkAffineRegistration(
        patch_size=args.patch_size,
        overlap=args.overlap,
        motion_type=args.motion_type,
    )
    registered_img, results = registrar.register(template_img, target_img)

    n_ok = sum(r.converged for r in results)
    print(f"Fitted {n_ok}/{len(results)} patches successfully.")
    cv2.imwrite(args.output, registered_img)
    print(f"Saved registered image to {args.output}")
