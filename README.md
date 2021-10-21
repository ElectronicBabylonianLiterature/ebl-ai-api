# ebl-ai-api
![Build Status](https://github.com/ElectronicBabylonianLiterature/ebl-ai-api/workflows/CI/badge.svg?branch=main)

# Ebl Ai Api
Server deploying a deep learning model for inference on detecting bounding boxes on cuneiform sign tablets
For **training + data/weights** please refer to [cuneiform-sign-detection repo](https://github.com/ElectronicBabylonianLiterature/cuneiform-sign-detection)


## Table of contents

* [Setup](#setup)
* [Codestyle](#codestyle)
* [Running the tests](#running-the-tests)
* [Running the application](#running-the-application)
* [Acknowledgements](#acknowledgements)

## Setup

Requirements:

* sudo apt-get install ffmpeg libsm6 libxext6  -y  
  (may be needed for open-cv python)


* Python 3.9

pip3 install -r requirements

Run `check_installation.py` to check pytorch, mmcv, mmdet and mmocr installation.

### Model
- Using [FCENet](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/fcenet/README.md) (CVPR'2021)
- FCENet implementation: [MMOCR](https://github.com/open-mmlab/mmocr)
- [FCENET with deconvolutions](https://mmocr.readthedocs.io/en/latest/textdet_models.html#id5) has slightly better performance but can't be used on cpu during inference
- [FCENET without deconvolutions](https://mmocr.readthedocs.io/en/latest/textdet_models.html#id6) is used in our production server [ebl-ai-api](https://github.com/ElectronicBabylonianLiterature/ebl-ai-api)


## Running the tests
- Use command `black ebl_ai_api` to format code.
- Use command `flake8` to linting.
- Use command `pytest` to run all tests.
- Use command `pyre` for type-checking.

## Running the server
`waitress-serve --port=8000 --call ebl.app:get_app`

## Acknowledgements
## Acknowledgements
- FCENET [https://arxiv.org/abs/2104.10442](https://arxiv.org/abs/2104.10442)
- Using [https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/fcenet/README.md](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/fcenet/README.md) (CVPR'2021)
- MMOCR [https://github.com/open-mmlab/mmocr](https://github.com/open-mmlab/mmocr)
- Deep learning of cuneiform sign detection with weak supervision using transliteration alignment [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0243039](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0243039)
- Synthetic Cuneiform Dataset (2000 Tablets) from [https://github.com/cdli-gh/Cuneiform-OCR](https://github.com/cdli-gh/Cuneiform-OCR)
- Annotated Tablets (75 Tablets) [https://compvis.github.io/cuneiform-sign-detection-dataset/](https://compvis.github.io/cuneiform-sign-detection-dataset/)