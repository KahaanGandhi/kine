# Copyright (C) 2026 Marianna Foschi, Antonio Fuentes, Brandon Zhao
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compact NUFFT operators for the static imaging interface."""

import numpy as np

import jax
import jax.numpy as jnp
from flax import struct

try:
    from jax_finufft import nufft2
except ImportError:
    nufft2 = None


def _validate_nufft_inputs(uv, prior, eps):
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
    """NUFFT operator to predict visibilities from one static image.

    Stores the (u, v) coordinates and pixel corrections instead of a dense
    Fourier matrix A. The first array axis indexes the baseline legs of each
    data product: one for an ordinary visibility (e.g., visI), three for a
    triangle closure, or four for a quadrangle closure.

    ``shape`` gives the image dimensions and ``eps`` the requested NUFFT accuracy.
    Both are fixed when JAX compiles the transform, so no global image size is needed.
    """

    u: jax.Array
    v: jax.Array
    factor: jax.Array
    shape: tuple[int, int] = struct.field(pytree_node=False)
    eps: float = struct.field(pytree_node=False, default=1e-6)

    @classmethod
    def from_uv(cls, uv, prior, eps=1e-6):
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

    def __call__(self, image):
        """Predict complex visibilities for every UV point and baseline leg."""
        # Restore flat pixel values to a (row, column) image
        image = jnp.asarray(image)
        if image.size != self.shape[0] * self.shape[1]:
            raise ValueError("The image size does not match the NUFFT operator")
        image = image.reshape(self.shape).astype(jnp.result_type(image.dtype, 1j))
        if self.u.shape[-1] == 0:
            return jnp.zeros(self.u.shape, dtype=image.dtype)
        # Image axes are (y, x), so pass (v, u), as Trainer's video NUFFT does
        def sample_leg(u, v, pixel_factor):
            return nufft2(image, v, u, eps=self.eps) * pixel_factor
        # Use the same candidate image for each leg; output shape is (nlegs, nvis)
        return jax.vmap(sample_leg)(
            self.u.astype(image.real.dtype),
            self.v.astype(image.real.dtype),
            self.factor.astype(image.dtype)
        )
