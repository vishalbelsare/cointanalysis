# coding: utf-8

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

description = """Python library for cointegration analysis."""


setup(
    name='cointanalysis',
    version='0.1.0',
    description=description,
    long_description=description,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    author='Shota Imaki',
    author_email='shota.imaki@icloud.com',
    maintainer='Shota Imaki',
    maintainer_email='shota.imaki@icloud.com',
    url='https://github.com/simaki/cointanalysis',
    packages=['cointanalysis'],
    license=license,
    classifiers=[
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Office/Business :: Financial :: Investment',
    ],
)
