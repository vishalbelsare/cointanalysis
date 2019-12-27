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
    version='0.0.1',
    description=description,
    long_description=description,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    author='Shota Imaki',
    author_email='shota.imaki@icloud.com',
    maintainer='Shota Imaki',
    maintainer_email='shota.imaki@icloud.com',
    url='https://github.com/simaki/cointanalysis',
    packages=find_packages(include=["coint", "coint.*"]),
    license=license,
    classifiers=[
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Topic :: Office/Business :: Financial :: Investment',
    ],
)
