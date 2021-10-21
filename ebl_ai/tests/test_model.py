import numpy as np
from PIL import Image

from ebl_ai.app import Model
from ebl_ai.model import BoundingBoxesPrediction

CONFIG_FILE = "model/fcenet_no_dcvn.py"
CHECKPOINT = "model/checkpoint.pth"


TEST_IMAGE_PATH = "ebl_ai/tests/test_image.jpg"


def test_model_predictions():
    model = Model(configFile=CONFIG_FILE, checkpoint=CHECKPOINT)

    predictions = model.predict(TEST_IMAGE_PATH)

    assert isinstance(predictions[0], BoundingBoxesPrediction)
    assert len(predictions) > 1

    model.show_result(TEST_IMAGE_PATH, "./test_image_prediction.jpg", False)

    img = Image.open(TEST_IMAGE_PATH)
    predictions = model.predict(np.asarray(img))
    assert isinstance(predictions[0], BoundingBoxesPrediction)
    assert len(predictions) > 1
