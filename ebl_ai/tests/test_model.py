import os

import numpy as np
from PIL import Image

from ebl_ai.app import Model
from ebl_ai.model import BoundingBoxesPrediction

CONFIG_FILE = "model/cpu/fcenet.py"
CHECKPOINT = "model/cpu/fcenet_r50_fpn_1500e_icdar2015-d435c061.pth"
TEST_IMAGE_PATH = "ebl_ai/tests/test_image.jpeg"


def test_model_predictions_from_path():
    model = Model(configFile=CONFIG_FILE, checkpoint=CHECKPOINT)

    predictions = model.predict(TEST_IMAGE_PATH)
    assert isinstance(predictions[0], BoundingBoxesPrediction)
    assert len(predictions) > 1


def test_model_predictions_from_np_array():
    model = Model(configFile=CONFIG_FILE, checkpoint=CHECKPOINT)

    img = Image.open(TEST_IMAGE_PATH)
    predictions = model.predict(np.asarray(img))
    assert isinstance(predictions[0], BoundingBoxesPrediction)
    assert len(predictions) > 1
