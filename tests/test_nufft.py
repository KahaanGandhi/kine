"""Numerical equivalence of the static direct and NUFFT interfaces."""

from collections import OrderedDict
from pathlib import Path
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import jax
import jax.numpy as jnp
import optax
import ehtim as eh
from ehtim.observing.obs_helpers import ftmatrix
from flax import linen as nn

from kine import fourier
from kine.fourier import NUFFT
from kine.model import NeuralField
from kine.obsdata import Obsdata
from kine.trainer import Trainer
from kine import utils as ut


class StaticNUFFTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'data/eht_m87_2017-04-10.uvfits'
        with warnings.catch_warnings(), ut.no_print():
            warnings.simplefilter('ignore', FutureWarning)
            cls.obs = Obsdata.load_uvfits(str(path)).avg_coherent(600)
        cls.prior = eh.image.make_square(cls.obs, 16, 160*eh.RADPERUAS)
        cls.image = jnp.asarray(np.random.default_rng(163).uniform(.001, .01, 256))

    def test_geometry(self):
        uv = np.array([[0., 0.], [2e9, -1.7e9], [-2e9, 1.7e9],
                       [1e9, 3e9], [1e9, -2e9], [2e9, -1.7e9]])
        predict = jax.jit(lambda pixels, A: A(pixels))
        for nx, ny in [(8, 8), (9, 7), (8, 7), (7, 8)]:
            with self.subTest(shape=(ny, nx)):
                prior = eh.image.Image(np.zeros((ny, nx)), 1e-11, 0., 0.)
                image = np.zeros((ny, nx), dtype=np.float32)
                image[1, 2] = .7
                operator = NUFFT.from_uv(uv, prior)
                actual = np.asarray(predict(jnp.asarray(image).ravel(), operator))[0]
                expected = ftmatrix(prior.psize, nx, ny, uv, pulse=prior.pulse) @ image.ravel()
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
                self.assertAlmostEqual(actual[0].real, .7, places=5)
                np.testing.assert_allclose(actual[1], actual[2].conj(), atol=2e-6)
                self.assertGreater(abs(actual[3]-actual[4]), .01)
                np.testing.assert_allclose(actual[1], actual[5], atol=2e-6)

    def test_targets_losses_and_gradients(self):
        for kind in ('vis', 'amp', 'logamp', 'bs', 'cphase', 'logcamp'):
            with self.subTest(product=kind):
                direct = Obsdata.get_data(self.obs, kind+'I', self.prior)
                fast = Obsdata.get_data(self.obs, kind+'I', self.prior, ttype='nufft')
                for a, b in zip(direct[:2], fast[:2]):
                    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
                def objective(image, product):
                    target, sigma, operator = product
                    return Trainer._loss_chi(ut.to_complex(image),
                        {'target': target, 'sigma': sigma, 'A': operator}, kind)
                evaluate = jax.jit(jax.value_and_grad(objective))
                a, ga = evaluate(self.image, direct)
                b, gb = evaluate(self.image, fast)
                np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)
                relative = np.linalg.norm(ga-gb)/np.linalg.norm(ga)
                self.assertLess(relative, 1e-4)

    def test_cached_data_tables(self):
        obs = self.obs.copy()
        with ut.no_print():
            obs.add_amp(debias=False)
            obs.add_bispec(count='max')
            obs.add_cphase(count='max')
            obs.add_logcamp(count='max', debias=False)
        for kind in ('amp', 'bs', 'cphase', 'logcamp'):
            with self.subTest(product=kind), ut.no_print():
                direct = Obsdata.get_data(obs, kind+'I', self.prior)
                fast = Obsdata.get_data(obs, kind+'I', self.prior, ttype='NUFFT')
                for a, b in zip(direct[:2], fast[:2]):
                    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)

    def test_no_dense_matrix_is_built(self):
        with patch('ehtim.observing.obs_helpers.ftmatrix', side_effect=AssertionError('dense allocation')):
            for kind in ('vis', 'amp', 'logamp', 'bs', 'cphase', 'logcamp'):
                _, _, operator = Obsdata.get_data(self.obs, kind+'I', self.prior, ttype='nufft')
                self.assertIsInstance(operator, NUFFT)
                self.assertEqual(operator.u.ndim, 2)
                self.assertEqual(operator.u.shape, operator.factor.shape)

    def test_stokes_channels(self):
        obs = self.obs.copy()
        # Known synthetic polarization on the measured baseline coordinates.
        for pol, value in [('Q', .1), ('U', .05j), ('V', -.02)]:
            obs.data[pol.lower()+'vis'] = value*obs.data['vis']
            obs.data[pol.lower()+'sigma'] = obs.data['sigma']
            for kind in ('vis', 'amp', 'logamp', 'bs', 'cphase', 'logcamp'):
                with self.subTest(product=kind+pol):
                    direct = Obsdata.get_data(obs, kind+pol, self.prior)
                    fast = Obsdata.get_data(obs, kind+pol, self.prior, ttype='nufft')
                    self.assertIsInstance(fast, list)
                    for a, b in zip(direct[:2], fast[:2]):
                        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
                    values = []
                    for target, sigma, A in (direct, fast):
                        values.append(Trainer._loss_chi(ut.to_complex(self.image),
                            {'target': target, 'sigma': sigma, 'A': A}, kind))
                    np.testing.assert_allclose(*values, rtol=1e-4, atol=1e-5)

    def test_optional_dependency_and_scope(self):
        with patch.object(fourier, 'nufft2', None):
            # The default path remains usable without the optional backend.
            self.assertEqual(len(Obsdata.get_data(self.obs, 'visI', self.prior)), 3)
            with self.assertRaisesRegex(ImportError, 'jax-finufft'):
                Obsdata.get_data(self.obs, 'visI', self.prior, ttype='nufft')
        with self.assertRaisesRegex(NotImplementedError, 'one Obsdata'):
            Obsdata.get_data([self.obs], 'visI', self.prior, ttype='nufft')
        with self.assertRaises(NotImplementedError):
            Obsdata.get_data(self.obs, 'mbreve', self.prior, ttype='nufft')
        for eps in (0., -1., np.nan):
            with self.subTest(eps=eps), self.assertRaises(ValueError):
                NUFFT.from_uv(np.zeros((1, 2)), self.prior, eps=eps)
        with self.assertRaises(ValueError):
            NUFFT.from_uv(np.zeros((2, 3)), self.prior)
        with self.assertRaises(ValueError):
            NUFFT.from_uv(np.array([[1+1j, 0]]), self.prior)
        operator = NUFFT.from_uv(np.empty((0, 2)), self.prior)
        self.assertEqual(operator(self.image).shape, (1, 0))
        with self.assertRaisesRegex(ValueError, 'image size'):
            operator(jnp.ones(2))

    def test_float64_directional_derivative(self):
        with jax.experimental.enable_x64():
            uv = np.array([[0., 0.], [1e9, 2e9], [-2e9, .7e9]])
            prior = eh.image.Image(np.zeros((7, 9)), 1e-11, 0., 0.)
            operator = NUFFT.from_uv(uv, prior, eps=1e-10)
            rng = np.random.default_rng(17)
            image = jnp.asarray(rng.uniform(.001, .01, (7, 9)))
            direction = jnp.asarray(rng.normal(size=image.shape))
            direction /= jnp.linalg.norm(direction)
            def loss(pixels):
                return jnp.mean(jnp.abs(operator(pixels)-(.2+.03j))**2)
            grad = jax.grad(loss)(image)
            step = 1e-5
            finite_difference = (loss(image+step*direction)-loss(image-step*direction))/(2*step)
            derivative = jnp.sum(grad*direction)
            np.testing.assert_allclose(derivative, finite_difference, rtol=1e-6, atol=1e-9)

    def test_native_training_call(self):
        prior = eh.image.make_square(self.obs, 8, 160*eh.RADPERUAS)
        grid = ut.get_grid(8, 8)
        model = NeuralField(posenc_deg=(0, 0), depth=1, width=16,
                            activ=nn.gelu, outactiv=nn.softplus,
                            outshift=0, scaling_i=.01)
        variables = model.init(jax.random.PRNGKey(163), grid, train=True)
        initial = Trainer.create(apply_fn=model.apply, params=variables['params'],
            batch_stats=variables['batch_stats'], tx=optax.adamax(.003))
        images, losses = [], []
        for backend in ('direct', 'nufft'):
            target, sigma, A = Obsdata.get_data(self.obs, 'visI', prior, ttype=backend)
            data = {'visI': {'target': target, 'sigma': sigma, 'A': A}}
            state = initial
            history = []
            for _ in range(12):
                loss, _, image, state = Trainer.train_step(OrderedDict(
                    state=state, grid=grid, data=data, lcurve=None))
                history.append(float(loss))
            self.assertTrue(np.isfinite(history).all())
            self.assertLess(history[-1], history[0])
            self.assertEqual(int(state.step), 12)
            final, _ = model.apply({'params': state.params, 'batch_stats': state.batch_stats},
                                   grid, train=True, mutable=['batch_stats'])
            images.append(np.asarray(final))
            losses.append(history)
        np.testing.assert_allclose(losses[0], losses[1], rtol=1e-4, atol=1e-5)
        # BatchNorm removes pre-normalization biases; compare the fitted images.
        a, b = images
        self.assertLess(np.linalg.norm(a-b)/np.linalg.norm(a), 1e-4)


if __name__ == '__main__':
    unittest.main()
