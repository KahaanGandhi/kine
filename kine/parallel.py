# Copyright (C) 2026 Marianna Foschi, Antonio Fuentes, Brandon Zhao
# Added by Kahaan Gandhi
# SPDX-License-Identifier: GPL-3.0-or-later

"""Train ``visI`` NUFFT images across local GPUs."""

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .fourier import NUFFT


class NUFFTParallelTrainer:
    """Use several GPUs for one ``visI`` NUFFT training run.

    Each GPU receives different visibilities but evaluates the same image or
    image stack.
    Their losses and gradients are combined before each update. Prepare
    ``values`` on CPU if the full dataset does not fit on one GPU.

    Example::

        parallel = NUFFTParallelTrainer(data['visI'])
        loss, ldict, image, state = Trainer.train_step(
            OrderedDict(state=state, grid=grid, data=data, lcurve=None),
            parallel=parallel
        )
    """

    def __init__(self, values: dict,
                 devices: Sequence[jax.Device] | None = None) -> None:
        """Split one ``visI`` entry across the local GPUs once.

        ``values`` is the ``data['visI']`` dictionary, for either one image
        or several image frames.
        """
        # trainer.py also imports this class, avoid circular import
        from .trainer import Trainer
        if devices is None:
            devices = jax.local_devices(backend='gpu')
        self.devices = tuple(devices)
        if not self.devices:
            raise ValueError("At least one device is required")
        # Network and grid will be replicated, only measurements are split
        self.mesh = Mesh(np.array(self.devices), ('gpu',))
        self.replicated = NamedSharding(self.mesh, P())
        self.sharded = NamedSharding(self.mesh, P('gpu'))
        self._validate_inputs(values)
        self.data = self._shard_data(values)

        # Define the work each GPU does with its own visibility slice
        def step(state, grid, values, lcurve):
            # Each GPU makes the same image, then fits only its own uv rows
            def local_loss(params, batch_stats):
                loss, (updates, ldict, (image,)) = Trainer._loss_fn_red(
                    params, batch_stats, state.apply_fn, grid=grid,
                    data={'visI': values}, lcurve=lcurve
                )
                count = jnp.sum(values['padmask'])
                total = jax.lax.psum(count, 'gpu')
                # Weight local means by real row count to recover the full-data mean
                # This also counts the lcurve loss (total flux per frame) only once
                return loss * count / total, (updates, ldict, image, count, total)

            # Compute a gradient from each GPU's share of the visibilities
            (weighted_loss, (updates, ldict, image, count, total)), grads = \
                jax.value_and_grad(local_loss, has_aux=True)(
                    state.params, state.batch_stats
                )
            # All GPUs apply the same combined gradient to replicated parameters
            grads = jax.lax.psum(grads, 'gpu')
            state = state.apply_gradients(grads=grads)
            # The same image grid gives each GPU the same batch-normalization update
            if state.batch_stats is not None:
                state = state.replace(batch_stats=updates[0]['batch_stats'])
            loss = jax.lax.psum(weighted_loss, 'gpu')
            # Combine the reported loss terms with the same row weights
            ldict = jax.tree_util.tree_map(
                lambda value: jax.lax.psum(value * count / total, 'gpu'), ldict
            )
            return loss, ldict, image, state

        # pmap runs the step on each GPU at the same time
        # Share state/grid/light curve, split data, and return one result
        self._step = jax.pmap(step, axis_name='gpu', devices=self.devices,
                              in_axes=(None, None, 0, None),
                              out_axes=(None, None, None, None))

    @staticmethod
    def _validate_inputs(values: dict) -> None:
        """Check once that the ``visI`` arrays can be split across GPUs."""
        operator = values['A']
        if (not isinstance(operator, NUFFT) or operator.u.ndim not in (2, 3)
                or operator.u.shape[-2] != 1):
            raise ValueError("Expected a visI NUFFT operator with one baseline leg")
        if operator.u.shape != operator.v.shape or operator.u.shape != operator.factor.shape:
            raise ValueError("NUFFT coordinates and pixel factors must have matching shapes")
        # Target and sigma have one row per image, without the baseline axis
        expected = (operator.u.shape[0], operator.u.shape[-1])
        if values['target'].shape != expected or values['sigma'].shape != expected:
            raise ValueError("Target and sigma must match the NUFFT operator")
        if operator.u.ndim == 3 and 'padmask' not in values:
            raise ValueError("Multiple images require a padding mask")
        if 'padmask' in values and values['padmask'].shape != expected:
            raise ValueError("Padding mask must match target and sigma")

    def _shard_data(self, values: dict) -> dict:
        """Put one equal-sized slice of the measurements on each GPU."""
        ndevices = len(self.devices)
        nvis = values['target'].shape[-1]
        if nvis < ndevices:
            raise ValueError("Need at least one visibility per device")
        width = (nvis + ndevices - 1) // ndevices
        extra = width * ndevices - nvis

        def shard(array, fill):
            host = np.asarray(jax.device_get(array))
            # Pad the last slice when needed and mask out the added rows
            if extra:
                pad = [(0, 0)] * (host.ndim - 1) + [(0, extra)]
                host = np.pad(host, pad, constant_values=fill)
            # Split the measurements into one equal-sized slice per GPU
            # Stack the slices so JAX places each one on a different GPU
            return jax.device_put(
                np.stack(np.split(host, ndevices, axis=-1)), self.sharded
            )

        operator = values['A']
        # Split all fields at the same rows so each GPU gets matching measurements
        # Padded rows are masked, with factor=1 to avoid undefined gradients
        sharded = NUFFT(shard(operator.u, 0), shard(operator.v, 0),
                        shard(operator.factor, 1), operator.shape, operator.eps)
        mask = values['padmask'] if 'padmask' in values else \
            np.ones(values['target'].shape, dtype=bool)
        return {'target': shard(values['target'], 0),
                'sigma': shard(values['sigma'], 1),
                'padmask': shard(mask, False), 'A': sharded}

    def train_step(self, kwargs: dict) -> tuple[jax.Array, dict, jax.Array, Any]:
        """Run one parallel update and return the usual training outputs."""
        state = jax.device_put(kwargs['state'], self.replicated)
        grid = jax.device_put(kwargs['grid'], self.replicated)
        lcurve = kwargs.get('lcurve')
        if lcurve is not None:
            lcurve = jax.device_put(lcurve, self.replicated)
        return self._step(state, grid, self.data, lcurve)
