# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

setup(
    name="ovisplice",
    version="1.0.0",
    description="Dual-species multi-tissue deep learning and Ruminant SpliceMap framework for pre-mRNA splicing",
    url="https://github.com/Ruminant-Genomics/OviSplice",
    license="MIT",
    packages=find_packages(include=["ovisplice", "ovisplice.*"]),
    py_modules=["ovisplice_predict"],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "openpyxl>=3.1",
    ],
    entry_points={
        "console_scripts": [
            "ovisplice=ovisplice_predict:main",
        ],
    },
)
