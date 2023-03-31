#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os

from setuptools import find_packages, setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, library_paths, include_paths

if 'CXX' not in os.environ:
    os.environ['CXX'] = 'g++'

def TinyCUDAExtension(name, sources, *args, **kwargs):
    library_dirs = kwargs.get('library_dirs', [])
    library_dirs += library_paths(cuda=True)
    kwargs['library_dirs'] = library_dirs

    libraries = kwargs.get('libraries', [])
    libraries.append('cudart')
    kwargs['libraries'] = libraries

    include_dirs = kwargs.get('include_dirs', [])

    include_dirs += include_paths(cuda=True)
    kwargs['include_dirs'] = include_dirs

    kwargs['language'] = 'c++'

    return Extension(name, sources, *args, **kwargs)

# Package meta-data.
NAME = 'collcache'
DESCRIPTION = 'A high-performance GPU-based graph sampler for deep graph learning application'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '3.0.0'

# What packages are required for this module to be executed?
REQUIRED = [
    # 'cffi>=1.4.0',
]


# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except OSError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

# Where the magic happens:

cxx_flags = [
    '-std=c++17', '-g',
    # '-fopt-info',
    '-fPIC',
    '-Ofast',
    # '-DPIPELINE',
    # '-O0',
    '-Wall', '-fopenmp', '-march=native',
    '-D_GLIBCXX_USE_CXX11_ABI=0'
]
cuda_flags = [
    '-Wno-deprecated-gpu-targets',
    '-std=c++17',
    '-g',
    # '-G',
    #  '--ptxas-options=-v',
    #  '-DPIPELINE',
    # '-Xptxas', '-dlcm=cv', # cache volatile   
    # '-DSXN_NAIVE_HASHMAP',
    '--compiler-options', "'-fPIC'",
    # '-gencode=arch=compute_35,code=sm_35',  # K40m
    '-gencode=arch=compute_70,code=sm_70',  # V100
    '-gencode=arch=compute_80,code=sm_80',  # A100
]

setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(exclude=('tests',)),
    install_requires=REQUIRED,
    include_package_data=True,
    license='Apache',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Operating System :: POSIX :: Linux'
    ],
    ext_modules=[
        TinyCUDAExtension(
            name='collcache.common.c_lib',
            sources=[
                'collcache/common/operation.cc',
            ],
            include_dirs=[
                os.path.join(here, '..'),
              ],
            libraries=['cudart', 'cusparse', 'coll_cache'],
            library_dirs=[
                os.path.join(here, '../build'),
            ],
            extra_link_args=['-Wl,--version-script=collcache.lds', '-fopenmp', '-Wl,-rpath=' + os.path.join(here, '../build')],
            # these custom march may should be remove and merged
            extra_compile_args={
                'cxx': cxx_flags,
                'nvcc': cuda_flags
            }),
        CUDAExtension(
            name='collcache.torch.c_lib',
            sources=[
                'collcache/torch/adapter.cc',
            ],
            include_dirs=[
                os.path.join(here, '..'),
              ],
            libraries=['cusparse', 'coll_cache'],
            library_dirs=[
              os.path.join(here, '../build')
            ],
            extra_link_args=['-Wl,--version-script=collcache.lds', '-fopenmp'],
            # these custom march may should be remove and merged
            extra_compile_args={
                'cxx': cxx_flags,
                'nvcc': cuda_flags
            }),
    ],
    # $ setup.py publish support.
    cmdclass={
        'build_ext': BuildExtension
    },
    # cffi is required for PyTorch
    # If cffi is specified in setup_requires, it will need libffi to be installed on the machine,
    # which is undesirable.  Luckily, `install` action will install cffi before executing build,
    # so it's only necessary for `build*` or `bdist*` actions.
    setup_requires=REQUIRED
)
