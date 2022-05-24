try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open("README.md") as f:
    longdesc = f.read()

version = "0.0.3"

config = {
    "description": "OG-UK",
    "long_description": longdesc,
    "url": "https://github.com/PSLmodels/OG-UK/",
    "download_url": "https://github.com/PLSmodels/OG-UK/",
    "version": version,
    "license": "CC0 1.0 Universal public domain dedication",
    "packages": ["oguk"],
    "include_package_data": True,
    "name": "oguk",
    "install_requires": [
        "openfisca-uk>=0.21.0",
        "ogcore>=0.8.0",
    ],
    "package_data": {"oguk": ["data/*"]},
    "classifiers": [
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: CC0 1.0 Universal public domain dedication",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    "tests_require": ["pytest"],
}

setup(**config)
