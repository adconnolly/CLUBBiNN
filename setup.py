#!/usr/bin/env python

from setuptools import setup, find_packages

description = "ML optimisation framework for subgrid forcing"
version = "0.0.1"

setup(
    name="subgrid_parameterization",
    version=version,
    description=description,
    url="https://github.com/m2lines/Universal_Parameterization",
    author="Chris Pedersen, Laure Zanna, Pavel Perezhogin, Adam Subel",
    author_email="c.pedersen@nyu.edu",
    packages=find_packages(),
)
