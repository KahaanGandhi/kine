# Copyright (C) 2026 Marianna Foschi, Antonio Fuentes, Brandon Zhao
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compact NUFFT operators for the static imaging interface."""

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
from ehtim import const_def as ehc
from ehtim.imaging import imager_utils as ehu
from ehtim.observing import obs_helpers as obsh

try:
    from jax_finufft import nufft2
except ImportError:
    nufft2 = None


@struct.dataclass
class NUFFT:
    """JAX-compatible coordinates and pixel factors, replacing dense ``A``.

    ``shape`` and ``eps`` are static under JIT. No global image size is needed.
    Each leading entry holds one baseline leg of a data product.
    """

    u: jax.Array
    v: jax.Array
    factor: jax.Array
    shape: tuple[int, int] = struct.field(pytree_node=False)
    eps: float = struct.field(pytree_node=False, default=1e-6)

    @classmethod
    def from_uv(cls, uv, prior, eps=1e-6):
        """Prepare one or more ``(nvis, 2)`` arrays, in wavelengths."""
        if nufft2 is None:
            raise ImportError("ttype='nufft' requires jax-finufft; install it with pip install jax-finufft")
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("nufft_eps must be finite and positive")
        uv = np.asarray(uv)
        if uv.ndim == 2:
            uv = uv[None, ...]
        if (uv.ndim != 3 or uv.shape[-1] != 2
                or not np.isrealobj(uv) or not np.isfinite(uv).all()):
            raise ValueError("uv must have shape (nvis, 2) or (nlegs, nvis, 2), with finite real values")
        if prior.xdim < 1 or prior.ydim < 1 or not np.isfinite(prior.psize) or prior.psize <= 0:
            raise ValueError("The prior must have positive image dimensions and pixel size")
        scaled = 2 * np.pi * prior.psize * uv
        # Match ehtim.ftmatrix's descending pixel axes, including odd grids.
        shift = np.array([1 - prior.xdim % 2, 1 - prior.ydim % 2]) / 2
        phase = np.exp(-1j * np.sum(scaled * shift, axis=-1))
        pulse = np.array([
            prior.pulse(2*np.pi*u, 2*np.pi*v, prior.psize, dom='F')
            for u, v in uv.reshape(-1, 2)
        ]).reshape(uv.shape[:-1])
        return cls(jnp.asarray(scaled[..., 0]), jnp.asarray(scaled[..., 1]),
                   jnp.asarray(phase * pulse), (prior.ydim, prior.xdim), float(eps))

    def __call__(self, image):
        """Return ``(nlegs, nvis)`` predictions for a flattened static image."""
        image = jnp.asarray(image)
        if image.size != self.shape[0] * self.shape[1]:
            raise ValueError("The image size does not match the NUFFT operator")
        image = image.reshape(self.shape).astype(jnp.result_type(image.dtype, 1j))
        if self.u.shape[-1] == 0:
            return jnp.zeros(self.u.shape, dtype=image.dtype)
        def sample(u, v, factor):
            return nufft2(image, v, u, eps=self.eps) * factor
        return jax.vmap(sample)(self.u.astype(image.real.dtype),
                                self.v.astype(image.real.dtype),
                                self.factor.astype(image.dtype))


def static_nufft_data(obs, dtype, prior, eps=1e-6):
    """Use the same data selection and noise conventions as the direct path."""
    if isinstance(obs, list):
        raise NotImplementedError("ttype='nufft' supports one Obsdata; use get_data_nfft for videos")
    if dtype not in {kind + pol for kind in ('vis', 'amp', 'logamp', 'bs', 'cphase', 'logcamp')
                     for pol in ('I', 'Q', 'U', 'V')}:
        raise NotImplementedError("ttype='nufft' supports static vis, amp, logamp, bs, cphase and logcamp")
    kind, pol = dtype[:-1], dtype[-1]
    vtype = ehc.vis_poldict[pol]
    if kind in ('vis', 'amp', 'logamp'):
        fields = ['t1', 't2', 'u', 'v', vtype, ehc.amp_poldict[pol], ehc.sig_poldict[pol]]
        cached = obs.amp if kind != 'vis' and pol == 'I' else None
        if cached is not None and len(cached):
            if not isinstance(cached, (np.ndarray, np.recarray)):
                raise TypeError("The pre-computed amplitude table must be a numpy record array")
            rows = cached
        else:
            rows = obs.unpack(fields, debias=False)
        uv, vis, amp, sigma = ehu.apply_systematic_noise_snrcut(rows, 0., 0., pol)
        target = vis if kind == 'vis' else amp
    else:
        # Retain cached tables and the same minimal closure sets as ehtim.
        if kind == 'bs':
            rows = obs.bispec
            if rows is None or len(rows) == 0 or pol != 'I':
                rows = obs.bispectra(mode='all', vtype=vtype, count='min', snrcut=0.)
            else:
                rows = obsh.reduce_tri_minimal(obs, rows)
            value, error, legs = 'bispec', 'sigmab', 3
        elif kind == 'cphase':
            rows = obs.cphase
            if rows is None or len(rows) == 0 or pol != 'I':
                rows = obs.c_phases(mode='all', vtype=vtype, count='min', uv_min=False, snrcut=0.)
            else:
                rows = obsh.reduce_tri_minimal(obs, rows)
            value, error, legs = 'cphase', 'sigmacp', 3
        else:
            rows = obs.logcamp
            if rows is None or len(rows) == 0 or pol != 'I':
                rows = obs.c_amplitudes(mode='all', count='min', vtype=vtype,
                                        ctype='logcamp', debias=False, snrcut=0.)
            else:
                rows = obsh.reduce_quad_minimal(obs, rows, ctype='logcamp')
            value, error, legs = 'camp', 'sigmaca', 4
        target, sigma = rows[value], rows[error]
        uv = np.stack([np.column_stack([rows[f'u{i}'], rows[f'v{i}']])
                       for i in range(1, legs+1)])
    operator = NUFFT.from_uv(uv, prior, eps)
    return [jnp.asarray(target).reshape(1, -1),
            jnp.asarray(sigma).reshape(1, -1), operator]
