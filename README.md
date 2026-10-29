# kine

`kine` is a Python package for video reconstruction of variable and sparse radio-interferometric data, from horizon-scale supermassive black holes to relativistic jets and more. It models the brightness distribution of the observed source in space, time, and frequency through a fully unsupervised neural field, parametrized by a coordinate-based neural network.

Built on [`JAX`](https://jax.readthedocs.io/) and [`Flax`](https://flax.readthedocs.io/), ``kine`` leverages GPU-accelerated automatic differentiation and JIT compilation for fast training. It uses and complements the [`eht-imaging`](https://github.com/achael/eht-imaging) library for VLBI data handling.

### Imaging modes

`kine` can be used for the following imaging tasks:

- **Static imaging**: reconstruct an image of the source from a single VLBI observation.
- **Dynamic imaging**: reconstruct a video of the source from a single VLBI observation.
- **Multi-epoch imaging**: reconstruct a video of the source across multiple observations spanning days to years.
- **Spectral imaging**: reconstruct the spectral dependence of the source across multiple frequencies.

### Available features

`kine` currently supports:

- **Full polarimetric** video and image reconstruction (Stokes I, Q, U, V).
- **Static + dynamic decomposition**: in dynamic mode, separate persistent and time-variable source structure.
- **Simultaneous gain fitting**: amplitude and phase telescope gains optimized jointly with the image.
- **GPU-based NUFFT**: Non-Uniform Fast Fourier Transform for direct visibility computation.
- **Multiple data products**: visibility amplitudes, closure phases, closure amplitudes, bispectra, and complex polarization ratios.

Coming Soon:

- Multi-epoch + multi-frequency imaging
- Scalability to large datasets

### Static and multi-epoch imaging with NUFFT

For a single observation, use `ttype='nufft'` to compute visibilities
without storing a dense Fourier matrix:

```python
from kine.obsdata import Obsdata

target, sigma, A = Obsdata.get_data(obs, 'visI', improxy, ttype='nufft')
data = {'visI': {'target': target, 'sigma': sigma, 'A': A}}
```

The training call stays the same and the direct transform remains the default.
This option requires `jax-finufft`. See the
[static NUFFT example](notebooks/example_static_nufft.ipynb) for a complete run.

For multiple observations, pass their list to the same function. Each frame
uses its own measured uv coordinates. Since observations can have different
numbers of measurements, Kine pads the shorter arrays to the same length,
and `padmask` omits those entries from the loss:

```python
target, sigma, A, padmask = Obsdata.get_data(obslist, 'visI', improxy, ttype='nufft')
data = {'visI': {'target': target, 'sigma': sigma, 'A': A, 'padmask': padmask}}
```

## Documentation

Full documentation is available at (https://mariannafoschi.github.io/kine/) and includes installation instructions, simple example scripts to get started, a complete user guide to imaging with kine, a parameter overview, and API documentation.

A full description of the imaging algorithm and extensive reconstruction validation tests are presented in the publications:

1. _[Main algorithm, Muti-epoch imaging, Static Imaging]_ : Foschi M., Zhao B., Fuentes A. et al. "Video reconstruction of variable VLBI observations with neural fields". Nature (2026).
2. _[Static + Dynamic decomposition, EHT SgrA* pipeline]_ : Fuentes A., Foschi M. et al. "Validation of horizon-scale Sagittarius A* video reconstructions with kine". Under review (2026).

## Developers

`kine` is developed and maintained by:

 - Marianna Foschi (foschimarianna @ gmail . com)
 - Antonio Fuentes (antoniofuentesfdez @ gmail . com)
 - Brandon Zhao (byzhao @ caltech . edu)

If you would like support for using kine in your project or find an issue in the code please contact Marianna. If you would like support for installing the code on your machine please reach out to Brandon.

 ## Citation

 If you use `kine` in your publication, please cite:

1. Foschi M., Zhao B., Fuentes A. et al. "Video reconstruction of variable VLBI observations with neural fields". Nature (2026).
2. Fuentes A., Foschi M. et al. "Validation of horizon-scale Sagittarius A* video reconstructions with kine". Under review (2026).



![kine video reconstruction from EHT-like data](docs/images/kine_eht.gif)

