# ebl-ai-api

## Table of contents

* [Setup](#setup)
* [Codestyle](#codestyle)
* [Running the tests](#running-the-tests)
* [Running the application](#running-the-application)
* [Acknowledgements](#acknowledgements)

## Setup

Requirements:

* Python 3.9

pip3 install -r requirements.txt

CPU version. `fcenet_dcvn.py` needs GPU
requirements have to be changed (Pytorch, mmcv-full and mmocr)

## Codestyle

Use [Black](https://black.readthedocs.io/en/stable/) codestyle and
[PEP8 naming conventions](https://www.python.org/dev/peps/pep-0008/#naming-conventions).
Line length is 88, and bugbear B950 is used instead of E501.
PEP8 checks should be enabled in PyCharm, but E501, E203, and E231 should be
disabled.

Use type hints in new code and add the to old code when making changes.

## Acknowledgements
