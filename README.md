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

CPU version. `fcenet_dcvn.py` needs GPU, so
requirements have to be changed (Pytorch, mmcv-full and mmocr)

### Model
- Using [FCENet](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/fcenet/README.md) (CVPR'2021)
- Training [Cuneiform Sign Detection](https://github.com/ElectronicBabylonianLiterature/cuneiform-sign-detection) (Checkpoints file linked here)

## Codestyle

Use [Black](https://black.readthedocs.io/en/stable/) codestyle and
[PEP8 naming conventions](https://www.python.org/dev/peps/pep-0008/#naming-conventions).

Use type hints in new code and add the to old code when making changes.

## Running the tests
`pytest`

## Running the server
`waitress-serve --port=8000 --call ebl.app:get_app`

## Acknowledgements
- Using [FCENet](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/fcenet/README.md) (CVPR'2021)
- MMOCR [MMOCR](https://github.com/open-mmlab/mmocr)
