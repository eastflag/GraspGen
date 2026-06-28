#!/bin/bash
# Installation script for GraspGen and pointnet2_ops with Blackwell GPU (RTX 5080) / CUDA 13.0 host compatibility

set -e  # Exit on any error

# 1. Clean and recreate virtual environment
echo "🧹 Deleting existing virtual environment (.venv)..."
rm -rf .venv

echo "🚀 Creating a new virtual environment with Python 3.10..."
uv venv --python 3.10 .venv
source .venv/bin/activate

# 2. Configure environment variables for Blackwell compilation (Compute Capability 12.0 / sm_120)
echo "🔧 Configuring environment variables for Blackwell (sm_120) compilation..."
export CC=/usr/bin/g++
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST="12.0"
export CUMM_CUDA_ARCH_LIST="12.0"

# 3. Install PyTorch 2.7.0 with CUDA 12.8 support
echo "📦 Installing PyTorch 2.7.0 with CUDA 12.8 support..."
uv pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# 4. Install PyG dependencies matching PyTorch 2.7.0 + CUDA 12.8
echo "📦 Installing PyG dependencies (torch-cluster, torch-scatter)..."
uv pip install torch-cluster torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# 5. Compile and install cumm from source to support sm_120 (Blackwell)
echo "📦 Compiling and installing cumm from source..."
rm -rf cumm
git clone --recursive https://github.com/FindDefinition/cumm.git
cd cumm
CUMM_DISABLE_JIT="1" uv pip install .
cd ..
rm -rf cumm

# 6. Compile and install spconv from source, patching it to support cumm 0.8.2+
echo "📦 Compiling and installing spconv from source..."
rm -rf spconv
git clone --recursive https://github.com/traveller59/spconv.git
cd spconv
sed -i 's/<0.8.0/<0.9.0/g' setup.py
SPCONV_DISABLE_JIT="1" uv pip install .
cd ..
rm -rf spconv

# 7. Install the GraspGen project dependencies
echo "📦 Installing GraspGen project..."
uv pip install -e .

# 8. Compile and install pointnet2_ops
echo "📦 Compiling and installing pointnet2_ops..."
cd pointnet2_ops && uv pip install --no-build-isolation . && cd ..

echo "🎉 All dependencies and pointnet2_ops installed successfully for CUDA 13.0 host / Blackwell GPU!"
