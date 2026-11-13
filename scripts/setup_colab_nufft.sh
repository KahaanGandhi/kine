#!/usr/bin/env bash
# Run in the current Colab GPU runtime, then restart the notebook session
set -euo pipefail
repo_dir="$1"
apt-get -qq update
apt-get -qq install -y libfftw3-dev
python_bin=python
# Install build tools and JAX before compiling jax-finufft
"$python_bin" -m pip install -q -c "$repo_dir/requirements.txt" \
    'jax[cuda13-local]' cmake ninja scikit-build-core nanobind setuptools-scm
cuda_arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '. ')
export CUDACXX=/usr/local/cuda/bin/nvcc
export PATH="/usr/local/cuda/bin:$PATH"
export CMAKE_BUILD_PARALLEL_LEVEL=2
# Rebuild even if a CPU-only copy is installed
"$python_bin" -m pip uninstall -y jax-finufft
# Pass the backend explicitly so pip applies the CUDA build settings
nufft_requirement=$(sed -n '/^jax-finufft @ /p' "$repo_dir/requirements.txt")
"$python_bin" -m pip install --no-build-isolation --no-cache-dir --no-deps \
    -Ccmake.define.JAX_FINUFFT_USE_CUDA=ON \
    "-Ccmake.define.CMAKE_CUDA_COMPILER=$CUDACXX" \
    "-Ccmake.define.CMAKE_CUDA_ARCHITECTURES=$cuda_arch" \
    "$nufft_requirement"
"$python_bin" -m pip install -q -r "$repo_dir/requirements-colab.txt"
"$python_bin" -m pip install -q --no-deps -e "$repo_dir"
"$python_bin" "$repo_dir/scripts/check_kine_cuda.py" --require-gpu --require-nufft
