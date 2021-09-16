# ebl-ai-api
![Build Status](https://github.com/ElectronicBabylonianLiterature/ebl-ai-api/workflows/CI/badge.svg?branch=main)
## Table of contents

* [Setup](#setup)
* [Codestyle](#codestyle)
* [Running the tests](#running-the-tests)
* [Running the application](#running-the-application)
* [Acknowledgements](#acknowledgements)

## Setup

Requirements:

* sudo apt-get install ffmpeg libsm6 libxext6  -y  
  (open-cv python dependencies)


* Python 3.9

`pip3 install pipenv`

`pipenv install --dev`

CPU version. 

`fcenet_dcvn.py` needs GPU, so requirements have to be changed (Pytorch)

run `check_installation.py` to check pytorch, mmcv, mmdet and mmocr installation.

Current issue on running `check_installation.py` is: https://github.com/open-mmlab/mmdetection/issues/3271

### Model
- Using [FCENet](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/fcenet/README.md) (CVPR'2021)
- Training [Cuneiform Sign Detection](https://github.com/ElectronicBabylonianLiterature/cuneiform-sign-detection) (Checkpoints file linked here)

## Codestyle

Use [Black](https://black.readthedocs.io/en/stable/) codestyle and
[PEP8 naming conventions](https://www.python.org/dev/peps/pep-0008/#naming-conventions).


Use type hints in new code and add the to old code when making changes.

## Running the tests
Use command `pytest` to run all tests.
Use command `pyre` for type-checking.

## Running the server
`waitress-serve --port=8000 --call ebl.app:get_app`

## Acknowledgements
- FCENET [https://arxiv.org/abs/2104.10442](https://arxiv.org/abs/2104.10442)
- Using [https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/fcenet/README.md](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/fcenet/README.md) (CVPR'2021)
- MMOCR [https://github.com/open-mmlab/mmocr](https://github.com/open-mmlab/mmocr)
- Deep learning of cuneiform sign detection with weak supervision using transliteration alignment [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0243039](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0243039)
- Synthetic Cuneiform Dataset (2000 Tablets) from [https://github.com/cdli-gh/Cuneiform-OCR](https://github.com/cdli-gh/Cuneiform-OCR)
- Annotated Tablets (75 Tablets) [https://compvis.github.io/cuneiform-sign-detection-dataset/](https://compvis.github.io/cuneiform-sign-detection-dataset/)
