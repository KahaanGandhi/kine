# Copyright (C) 2026 Marianna Foschi, Antonio Fuentes, Brandon Zhao
# Added by Kahaan Gandhi
# SPDX-License-Identifier: GPL-3.0-or-later

"""Sample measured visibilities for NUFFT training."""

from collections.abc import Iterator

import jax
import numpy as np

from .fourier import NUFFT


class NUFFTVisibilityBatcher:
    """Keep ``visI`` data on CPU and send one batch per training step.

    Pass the ``visI`` entry returned by ``Obsdata.get_data`` or prepared
    directly from a table (e.g. HL Tau). Prepare that entry on CPU, so
    the full dataset need not be stored on GPU. For multiple epochs or
    bands, draw measurements from each image in proportion to its row
    count. The padding mask excludes filler rows from the loss.
    """

    def __init__(self, values: dict, batch_size: int, device: jax.Device,
                 seed: int = 0) -> None:
        """Store the measurements on CPU and set up sampling once.

        For several images, keep each frame's real rows separately and decide
        how many rows it contributes to each batch.
        """
        mask, counts = self._validate_inputs(values, batch_size)
        operator = values['A']
        # Read the arrays as NumPy so batches can be selected on CPU
        u = np.asarray(jax.device_get(operator.u))
        v = np.asarray(jax.device_get(operator.v))
        factor = np.asarray(jax.device_get(operator.factor))
        target = np.asarray(jax.device_get(values['target']))
        sigma = np.asarray(jax.device_get(values['sigma']))
        self.multiframe = operator.u.ndim == 3
        self.nvis = int(counts.sum())
        self.batch_size = int(batch_size)
        self.shape = operator.shape
        self.eps = operator.eps
        self.device = device
        self.rng = np.random.default_rng(seed)
        if self.multiframe:
            self.counts = counts
            self.frame_batch_sizes = self._split_batch(counts, self.batch_size)
            self.max_frame_batch = int(self.frame_batch_sizes.max())
            self.frames = []
            for frame in range(len(counts)):
                # Drop padded rows and keep each frame's fields in the same order
                valid = np.flatnonzero(mask[frame])
                if self.frame_batch_sizes[frame] < counts[frame]:
                    valid = valid[self.rng.permutation(len(valid))]
                self.frames.append({
                    'u': u[frame, 0, valid], 'v': v[frame, 0, valid],
                    'factor': factor[frame, 0, valid],
                    'target': target[frame, valid], 'sigma': sigma[frame, valid]
                })
        else:
            self.u, self.v, self.factor = u, v, factor
            self.target, self.sigma = target, sigma
            if batch_size < self.nvis:
                # Shuffle all fields together so each row keeps its uv and sigma
                order = self.rng.permutation(self.nvis)
                self.u = self.u[:, order]
                self.v = self.v[:, order]
                self.factor = self.factor[:, order]
                self.target = self.target[:, order]
                self.sigma = self.sigma[:, order]

    @staticmethod
    def _validate_inputs(values: dict, batch_size: int) -> tuple[np.ndarray | None, np.ndarray]:
        """Check once upon initialization that the NUFFT data can be sampled in batches.

        Return the padding mask and the number of real measurements per frame.
        """
        operator = values['A']
        if (not isinstance(operator, NUFFT) or operator.u.ndim not in (2, 3)
                or operator.u.shape[-2] != 1):
            raise ValueError("Expected a NUFFT operator with one visibility leg")
        if operator.v.shape != operator.u.shape or operator.factor.shape != operator.u.shape:
            raise ValueError("NUFFT coordinates and pixel factors must have matching shapes")
        # One image has shape (1, nvis) whereas several images add a frame axis
        if operator.u.ndim == 3:
            expected = (operator.u.shape[0], operator.u.shape[-1])
            if 'padmask' not in values:
                raise ValueError("Multi-frame visibility data requires a padding mask")
            mask = np.asarray(jax.device_get(values['padmask']))
            if (values['target'].shape != expected or values['sigma'].shape != expected
                    or mask.shape != expected or mask.dtype != np.bool_):
                raise ValueError("Multi-frame target, sigma and padmask must match the NUFFT operator")
            # Only unmasked rows count toward the frame size
            counts = mask.sum(axis=1)
        else:
            mask = None
            if (operator.u.shape[0] != 1 or values['target'].shape != operator.u.shape
                    or values['sigma'].shape != operator.u.shape or 'padmask' in values):
                raise ValueError("Static target and sigma must have shape (1, nvis)")
            counts = np.array([operator.u.shape[-1]])
        if (not isinstance(batch_size, (int, np.integer))
                or isinstance(batch_size, (bool, np.bool_))
                or not 1 <= batch_size <= counts.sum()):
            raise ValueError("Batch size must be an integer between 1 and the number of visibilities")
        if operator.u.ndim == 3 and batch_size < np.count_nonzero(counts):
            raise ValueError("Batch size must include at least one visibility per nonempty frame")
        return mask, counts

    @staticmethod
    def _split_batch(counts: np.ndarray, batch_size: int) -> np.ndarray:
        """Decide how many rows to draw from each frame per step.

        Use each frame's share of the full data, then round to whole rows so
        the returned counts add up to ``batch_size``.
        """
        shares = batch_size * counts / counts.sum()
        # Give every nonempty frame at least one row
        sizes = np.floor(shares).astype(int)
        sizes[(counts > 0) & (sizes == 0)] = 1
        # Add rows where rounding left a frame furthest below its share
        while sizes.sum() < batch_size:
            shortfall = np.where(sizes < counts, shares-sizes, -np.inf)
            frame = np.argmax(shortfall)
            sizes[frame] += 1
        # Remove extras if the one-row minimum made the batch too large
        while sizes.sum() > batch_size:
            excess = np.where(sizes > 1, sizes-shares, -np.inf)
            frame = np.argmax(excess)
            sizes[frame] -= 1
        # Return an array telling us how many visibilities to draw from each frame
        return sizes

    @staticmethod
    def _select_window(values: np.ndarray, start: int, size: int) -> np.ndarray:
        """Take ``size`` consecutive rows beginning at ``start``.

        If the window reaches the end, wrap around to the start.
        """
        if size == 0:
            return values[..., :0]
        end = start + size
        # Most windows stay within the array, so slicing avoids a copy
        if end <= values.shape[-1]:
            # ... keeps leading axes unchanged
            # start:end takes rows from start up to, but not including, end
            return values[..., start:end]
        return np.concatenate((values[..., start:],
                               values[..., :end-values.shape[-1]]), axis=-1)

    def sample(self) -> dict:
        """Draw the measured visibilities for one training step.

        Choose a starting row for each image, then return its selected NUFFT
        coordinates, targets and sigmas on the requested device.

        Partial batches use different visibilities each step, so their loss
        and gradient need not match a full-data step. For large batches,
        rounding changes each frame's allocation by only one row at most.
        """
        if self.multiframe:
            # Use fixed counts per frame so every training step has the same JAX shape
            starts = [0 if size == count else int(self.rng.integers(count))
                      for count, size in zip(self.counts, self.frame_batch_sizes)]
            return self._take_frames(starts, self.frame_batch_sizes)
        start = 0 if self.batch_size == self.nvis else int(self.rng.integers(self.nvis))
        return self._take(start, self.batch_size)

    def all_batches(self) -> Iterator[dict]:
        """Yield nonrandom chunks for a full-data evaluation.

        Every real visibility appears once. For several images, shorter
        frames are padded and masked after their rows run out.
        """
        if self.multiframe:
            # A frame can finish early, so later chunks leave its rows masked out
            steps = max((count + size - 1) // size for count, size in
                        zip(self.counts, self.frame_batch_sizes) if size)
            for step in range(steps):
                starts = [min(step*size, count)
                          for count, size in zip(self.counts, self.frame_batch_sizes)]
                sizes = [min(size, count-start)
                         for count, size, start in zip(self.counts, self.frame_batch_sizes, starts)]
                yield self._take_frames(starts, sizes)
            return
        for start in range(0, self.nvis, self.batch_size):
            yield self._take(start, min(self.batch_size, self.nvis-start))

    def _take(self, start: int, size: int) -> dict:
        """Build the data dictionary for one image's selected rows.

        Use the same window for the NUFFT coordinates, target and sigma,
        then move that batch to the requested device.
        """
        operator = NUFFT(
            jax.device_put(self._select_window(self.u, start, size), self.device),
            jax.device_put(self._select_window(self.v, start, size), self.device),
            jax.device_put(self._select_window(self.factor, start, size), self.device),
            self.shape,
            self.eps
        )
        return {
            'target': jax.device_put(self._select_window(self.target, start, size), self.device),
            'sigma': jax.device_put(self._select_window(self.sigma, start, size), self.device),
            'A': operator
        }

    def _take_frames(self, starts: list[int], sizes: list[int] | np.ndarray) -> dict:
        """Build one padded batch containing rows from several images.

        Select each frame's rows, mark padded rows with ``padmask``, then move
        the batch to the device.
        """
        # Pad to the largest batch so the operator has one array shape
        nframes, width = len(self.frames), self.max_frame_batch
        u = np.zeros((nframes, 1, width), dtype=self.frames[0]['u'].dtype)
        v = np.zeros_like(u)
        factor = np.ones((nframes, 1, width), dtype=self.frames[0]['factor'].dtype)
        target = np.zeros((nframes, width), dtype=self.frames[0]['target'].dtype)
        sigma = np.ones((nframes, width), dtype=self.frames[0]['sigma'].dtype)
        padmask = np.zeros((nframes, width), dtype=bool)
        # Fill the selected slots
        for frame, (start, size) in enumerate(zip(starts, sizes)):
            rows = self.frames[frame]
            u[frame, 0, :size] = self._select_window(rows['u'], start, size)
            v[frame, 0, :size] = self._select_window(rows['v'], start, size)
            factor[frame, 0, :size] = self._select_window(rows['factor'], start, size)
            target[frame, :size] = self._select_window(rows['target'], start, size)
            sigma[frame, :size] = self._select_window(rows['sigma'], start, size)
            padmask[frame, :size] = True
        # Move only this padded batch to the device for the NUFFT
        operator = NUFFT(jax.device_put(u, self.device),
                         jax.device_put(v, self.device),
                         jax.device_put(factor, self.device), self.shape, self.eps)
        batch = {'target': jax.device_put(target, self.device),
                 'sigma': jax.device_put(sigma, self.device), 'A': operator,
                 'padmask': jax.device_put(padmask, self.device)}
        return batch
