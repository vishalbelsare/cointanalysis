#!/bin/sh

python3 -m pytest --doctest-modules cointanalysis
python3 -m pytest --doctest-modules tests

python3 -m flake8 cointanalysis
python3 -m black --check cointanalysis || read -p "Run formatter? (y/N): " yn; [[ $yn = [yY] ]] && python3 -m black cointanalysis
python3 -m isort --check --force-single-line-imports cointanalysis || read -p "Run formatter? (y/N): " yn; [[ $yn = [yY] ]] && python3 -m isort --force-single-line-imports cointanalysis
