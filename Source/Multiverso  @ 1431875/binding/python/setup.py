from setuptools import setup
import os


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='multiverso-python',
      version='0.0.1',
      long_description=readme(),
      description="Multiverso is a parameter server framework for distributed"
      " machine learning. This package can leverage multiple machines and GPUs"
      " to speed up the python programs.",
      url='https://github.com/Microsoft/multiverso',
      author='Microsoft',
      license='MIT',
      packages=['multiverso', 'multiverso.theano_ext', 'multiverso.theano_ext.lasagne_ext'],
      # TODO: The lasagne on pypi is too old. multiverso need some functions in
      # lasagne-0.2 which is not released yet. Please replace the dev version
      # with the stable release later.
      dependency_links = ['https://github.com/Lasagne/Lasagne/tarball/master#egg=lasagne-0.2.dev1'],
      install_requires=["theano>=0.8.2", "lasagne>=0.2.dev1"],
      classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 2",
      ],
      zip_safe=False)
