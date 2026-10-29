# Copyright (C) 2026 Marianna Foschi, Antonio Fuentes, Brandon Zhao
# SPDX-License-Identifier: GPL-3.0-or-later

"""NUFFT operators for static and multi-frame imaging."""

from typing_extensions import Self

import numpy as np
import ehtim as eh

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from flax import struct

try:
    from jax_finufft import nufft2
except ImportError:
    nufft2 = None


def _validate_nufft_inputs(
        uv: np.ndarray,
        prior: eh.image.Image,
        eps: float
) -> None:
    """Validate the inputs before the first NUFFT call."""
    if nufft2 is None:
        raise ImportError("ttype='nufft' requires jax-finufft; install it with pip install jax-finufft")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("Tolerance for NUFFT computation must be finite and positive")
    if (uv.ndim != 3 or uv.shape[-1] != 2 or not np.isrealobj(uv) or not np.isfinite(uv).all()):
        raise ValueError("UV coordinates must have shape (nvis, 2) or (nlegs, nvis, 2), with finite real values")
    if not np.isfinite(prior.psize) or prior.psize <= 0:
        raise ValueError("Pixel size must be finite and positive")


@struct.dataclass
class NUFFT:
    """NUFFT operator to predict visibilities from one or more images.

    Stores the (u, v) coordinates and pixel corrections instead of a dense
    Fourier matrix A. The first array axis indexes the baseline legs of each
    data product: one for an ordinary visibility (e.g., visI), three for a
    triangle closure, or four for a quadrangle closure. For multiple frames,
    the frame axis comes before the baseline-leg axis.

    ``shape`` gives the image dimensions and ``eps`` the requested NUFFT accuracy.
    Both are fixed when JAX compiles the transform, so no global image size is needed.
    """

    u: jax.Array
    v: jax.Array
    factor: jax.Array
    shape: tuple[int, int] = struct.field(pytree_node=False)
    eps: float = struct.field(pytree_node=False, default=1e-6)

    @classmethod
    def from_uv(
            cls,
            uv: ArrayLike,
            prior: eh.image.Image,
            eps: float = 1e-6
    ) -> Self:
        """Prepare NUFFT sampling coordinates from UV points in wavelengths."""
        uv = np.asarray(uv)
        # Give ordinary visibilities the same leading baseline axis as closures
        if uv.ndim == 2:
            uv = uv[None, ...]
        _validate_nufft_inputs(uv, prior, eps)
        # Scale the observed UV coordinates for this image's pixel width,
        # following Obsdata.get_uvpoints
        scaled_uv = 2 * np.pi * prior.psize * uv
        # Align the NUFFT grid with ehtim.ftmatrix's pixel centers
        # Even axes need a half-pixel phase shift; odd axes have a pixel
        # at the origin
        pixel_shift = np.array([1 - prior.xdim % 2, 1 - prior.ydim % 2]) / 2
        phase_factor = np.exp(-1j * np.sum(scaled_uv * pixel_shift, axis=-1))
        # Account for each pixel's finite shape (triangular by default), as
        # ehtim.ftmatrix does through prior.pulse
        pulse_response = np.array([
            prior.pulse(2*np.pi*u, 2*np.pi*v, prior.psize, dom='F')
            for u, v in uv.reshape(-1, 2)
        ]).reshape(uv.shape[:-1])
        return cls(
            jnp.asarray(scaled_uv[..., 0]),
            jnp.asarray(scaled_uv[..., 1]),
            jnp.asarray(phase_factor * pulse_response),
            (prior.ydim, prior.xdim),
            float(eps)
        )

    @classmethod
    def stack(cls, operators: list[Self]) -> Self:
        """Pad and stack one operator per frame without building Fourier matrices."""
        # Frames can have different measurement counts; pad to the longest
        maxvis = max(op.u.shape[-1] for op in operators)
        def pad(values, fill=0):
            return jnp.stack([
                jnp.pad(value, ((0, 0), (0, maxvis-value.shape[-1])),
                        constant_values=fill)
                for value in values
            ])
        return cls(
            pad([op.u for op in operators]),
            pad([op.v for op in operators]),
            # Keep padded visibilities nonzero for log/phase losses
            pad([op.factor for op in operators], fill=1),
            operators[0].shape,
            operators[0].eps
        )

    def __call__(self, image: ArrayLike) -> jax.Array:
        """Predict one complex visibility per baseline leg and measurement.

        Return ``(nlegs, nvis)`` for one image or ``(nframes, nlegs, nvis)``
        for multiple frames.
        """
        image = jnp.asarray(image)
        nframes = self.u.shape[0] if self.u.ndim == 3 else 1
        if image.size != nframes * self.shape[0] * self.shape[1]:
            raise ValueError("The image size does not match the NUFFT operator")
        # Reshape flat pixel output to (frames, image_height, image_width)
        image = image.reshape((nframes,) + self.shape).astype(jnp.result_type(image.dtype, 1j))
        if self.u.shape[-1] == 0:
            return jnp.zeros(self.u.shape, dtype=image.dtype)
        # Image axes are (y, x), so pass (v, u), as Trainer's video NUFFT does
        def sample_frame(img, u, v, factor):
            # Sample every baseline leg in one transform, then restore its axis
            vis = nufft2(img, v.ravel().astype(img.real.dtype),
                         u.ravel().astype(img.real.dtype), eps=self.eps)
            return vis.reshape(u.shape) * factor.astype(img.dtype)
        if self.u.ndim == 2:
            return sample_frame(image[0], self.u, self.v, self.factor)
        return jax.vmap(sample_frame)(image, self.u, self.v, self.factor)
