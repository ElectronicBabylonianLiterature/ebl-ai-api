import io

import falcon
import numpy as np
import pytest
from PIL import Image
from falcon import testing

from ebl_ai.app import Model, create_app, BoundingBoxesPrediction

CONFIG_FILE = "../../model/cpu/fcenet.py"
CHECKPOINT = "../../model/cpu/fcenet_r50_fpn_1500e_icdar2015-d435c061.pth"

@pytest.fixture
def model():
    return Model(configFile=CONFIG_FILE,
                  checkpoint=CHECKPOINT)


@pytest.fixture
def client(model):
    api = create_app(model)
    return testing.TestClient(api)


def test_predictions_route(client):
    img = Image.open("./test_image.jpeg")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    result = client.simulate_post("/generate", body=byte_im, headers={'content-type': 'image/png'})
    assert result.status == falcon.HTTP_OK
    result_rectangle = result.json["boundaryResults"][0]
    assert isinstance(result_rectangle["top_left_x"], float)



def test_model_predictions_from_path(model):
    model = Model(configFile=CONFIG_FILE,
                  checkpoint=CHECKPOINT)

    predictions = model.predict("./test_image.jpeg")
    assert isinstance(predictions[0], BoundingBoxesPrediction)
    assert len(predictions) > 1

def test_model_predictions_from_np_array(model):
    model = Model(configFile=CONFIG_FILE,
                  checkpoint=CHECKPOINT)

    img = Image.open("./test_image.jpeg")
    predictions = model.predict(np.asarray(img))
    assert isinstance(predictions[0], BoundingBoxesPrediction)
    assert len(predictions) > 1