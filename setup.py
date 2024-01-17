"""
trainer package for training the models
"""

import os
from typing import List
from setuptools import setup
from setuptools import find_packages




NAME:str = "trainer"
AUTHOR:str = "hamzeh asadi"
VERSION:str = "0.1.0"
LICENSE:str = "MIT"
DESCRIPTION:str = "deep learning model training pipeline"
LONG_DESCRIPTION:str = open(os.path.join(os.getcwd(), "README.md")).read()
EMAIL:str = "hamzeasad@gmail.com"
REQUIER_PACKAGES:List = ["torch", "einops"]


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    packages=find_packages(),
    license=LICENSE,
    requires=REQUIER_PACKAGES
)

