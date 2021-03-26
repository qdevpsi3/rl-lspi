from setuptools import find_packages, setup

setup(name='lspi',
      version='0.0.1',
      install_requires=['gym', 'numpy', 'scikit-learn'],
      packages=find_packages('src'),
      package_dir={'': 'src'})
