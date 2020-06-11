#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
with open("requirements.txt") as rf:
    requirements = rf.read().splitlines()
    
setuptools.setup(
    name="bicleaner",
    version="0.14",
    install_requires=requirements,
    license="GNU General Public License v3.0",
    author="Prompsit Language Engineering",
    author_email="info@prompsit.com",
    maintainer="Marta Bañón",
    maintainer_email="mbanon@prompsit.com",
    description="Parallel corpus classifier, indicating the likelihood of a pair of sentences being mutual translations or not",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bitextor/bicleaner",
    packages=setuptools.find_packages(),
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Filters"
    ],
    project_urls={
        "Bicleaner on GitHub": "https://github.com/bitextor/bicleaner",
        "Prompsit Language Engineering": "http://www.prompsit.com",
        "Bicrawler & Bicleaner": "https://bicrawler.com",
        "Paracrawl": "https://paracrawl.eu/"
         },
    scripts=[
         "scripts/bicleaner-classify",
         "scripts/bicleaner-train",
         "scripts/bicleaner-classify-lite",
         "scripts/bicleaner-hardrules"
         ]
)
