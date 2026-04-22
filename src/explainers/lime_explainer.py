"""LIME for image classification – implemented from scratch.

Algorithm (follows Ribeiro et al. 2016):
  1. Segment the input image into superpixels.
  2. Sample binary masks over superpixels (presence / absence).
  3. For each mask, compose a perturbed image (absent superpixels → mean colour).
  4. Query the black-box model on all perturbed images.
  5. Weight each sample by its proximity to the all-ones mask (kernel).
  6. Fit a sparse linear model (Lasso / Ridge) on the weighted samples.
  7. Return the coefficient vector as a feature-importance heatmap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image
from skimage.segmentation import slic
from sklearn.linear_model import Ridge

from src.data.cifar10 import VIT_TRANSFORM, _VIT_MEAN, _VIT_STD


@dataclass
class LIMEResult:
    segments: np.ndarray          # (H, W) int array of superpixel labels
    coefficients: np.ndarray      # (n_segments,) linear-model weights
    heatmap: np.ndarray           # (H, W) float heatmap (coefficient per pixel)
    class_idx: int                # explained class index (CIFAR-10)
    runtime_seconds: float = 0.0
    extra: dict = field(default_factory=dict)


class LIMEImageExplainer:
    """Model-agnostic LIME for image classifiers.

    Parameters
    ----------
    n_samples:
        Number of perturbed neighbourhood samples to generate.
    n_segments:
        Approximate number of SLIC superpixels.
    kernel_width:
        Controls the locality of the exponential kernel.
        Smaller → tighter neighbourhood.
    alpha:
        Ridge regularisation strength for the surrogate model.
    seed:
        Random seed for reproducibility.
    batch_size:
        How many perturbed images to forward through the model at once.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        n_segments: int = 50,
        kernel_width: float = 0.25,
        alpha: float = 1.0,
        seed: int = 42,
        batch_size: int = 64,
    ) -> None:
        self.n_samples    = n_samples
        self.n_segments   = n_segments
        self.kernel_width = kernel_width
        self.alpha        = alpha
        self.seed         = seed
        self.batch_size   = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        image: np.ndarray,
        predict_fn,
        class_idx: int,
        device: str | torch.device = "cpu",
    ) -> LIMEResult:
        """Explain one prediction.

        Parameters
        ----------
        image:
            Raw uint8 RGB image of shape (H, W, 3), values in [0, 255].
        predict_fn:
            Callable that accepts a (N, C, H, W) float tensor (normalised,
            ViT-ready) and returns (N, 10) probability tensor.
        class_idx:
            CIFAR-10 class index to explain.
        device:
            Device used for model calls.
        """
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.seed)

        # 1. Superpixel segmentation
        segments = slic(
            image,
            n_segments=self.n_segments,
            compactness=10,
            sigma=1,
            start_label=0,
        )
        n_segs = segments.max() + 1

        # 2. Sample binary masks  (n_samples × n_segs)
        masks = rng.integers(0, 2, size=(self.n_samples, n_segs)).astype(np.float32)
        # Always include the original image (all-ones mask) as row 0
        masks[0] = 1.0

        # 3. Build perturbed images and preprocess them
        mean_colour = image.mean(axis=(0, 1))  # (3,)
        tensors = self._build_batch(image, segments, masks, mean_colour)  # (N, 3, 224, 224)

        # 4. Query the black-box model in mini-batches
        probs = self._query_model(tensors, predict_fn, device)  # (N, 10)
        labels = probs[:, class_idx]                            # (N,)

        # 5. Locality kernel: distance from all-ones mask in normalised Hamming space
        distances = self._distances(masks)   # (N,)
        weights   = self._kernel(distances)  # (N,)

        # 6. Weighted sparse linear surrogate
        ridge = Ridge(alpha=self.alpha)
        ridge.fit(masks, labels, sample_weight=weights)
        coeffs = ridge.coef_  # (n_segs,)

        # 7. Map coefficients back to pixel space
        heatmap = np.zeros(segments.shape, dtype=np.float32)
        for seg_id, coef in enumerate(coeffs):
            heatmap[segments == seg_id] = coef

        runtime = time.perf_counter() - t0
        return LIMEResult(
            segments=segments,
            coefficients=coeffs,
            heatmap=heatmap,
            class_idx=class_idx,
            runtime_seconds=runtime,
            extra={"n_segments_actual": n_segs, "n_samples": self.n_samples},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_batch(
        self,
        image: np.ndarray,
        segments: np.ndarray,
        masks: np.ndarray,
        mean_colour: np.ndarray,
    ) -> torch.Tensor:
        """Apply each binary mask to the image and return a batch tensor."""
        tensors = []
        pil_orig = Image.fromarray(image)
        for mask_row in masks:
            perturbed = image.copy()
            for seg_id, active in enumerate(mask_row):
                if active == 0:
                    perturbed[segments == seg_id] = mean_colour
            pil_img = Image.fromarray(perturbed)
            tensors.append(VIT_TRANSFORM(pil_img))
        return torch.stack(tensors)  # (N, 3, 224, 224)

    def _query_model(
        self,
        tensors: torch.Tensor,
        predict_fn,
        device,
    ) -> np.ndarray:
        results = []
        for i in range(0, len(tensors), self.batch_size):
            batch = tensors[i : i + self.batch_size].to(device)
            with torch.no_grad():
                probs = predict_fn(batch)
            results.append(probs.cpu().numpy())
        return np.concatenate(results, axis=0)  # (N, 10)

    def _distances(self, masks: np.ndarray) -> np.ndarray:
        """Cosine distance from each mask to the all-ones vector."""
        reference = np.ones(masks.shape[1], dtype=np.float32)
        dot   = masks @ reference                           # (N,)
        norms = np.linalg.norm(masks, axis=1) * np.linalg.norm(reference)
        norms = np.where(norms == 0, 1e-8, norms)
        cosine_sim = np.clip(dot / norms, -1.0, 1.0)       # avoids sqrt(negative) from float error
        return np.sqrt(2.0 * (1.0 - cosine_sim))           # cosine distance

    def _kernel(self, distances: np.ndarray) -> np.ndarray:
        return np.exp(-(distances ** 2) / (self.kernel_width ** 2))
