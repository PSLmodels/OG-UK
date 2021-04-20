try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open("README.md") as f:
    longdesc = f.read()

version = "0.0.0"

config = {
    "description": "UK Calibration for OG-MOD",
    "long_description": longdesc,
    "url": "https://github.com/PSLmodels/OG-UK-Calibration/",
    "download_url": "https://github.com/PLSmodels/OG-UK-Calibration/",
    "version": version,
    "license": "CC0 1.0 Universal public domain dedication",
    "packages": ["og_uk_calibrate"],
    "include_package_data": True,
    "name": "og_uk_calibrate",
    "install_requires": [],
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
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    "tests_require": ["pytest"],
}

setup(**config)
