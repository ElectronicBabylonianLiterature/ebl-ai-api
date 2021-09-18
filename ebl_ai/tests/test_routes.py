import io

import falcon
import pytest
from PIL import Image
from falcon import testing

from ebl_ai.app import Model, create_app


@pytest.fixture
def model():
    return Model(
        configFile="model/gpu/fcenet_dcvn.py",
        checkpoint="model/gpu/best_hmean-iou_hmean_epoch_200.pth",
    )


@pytest.fixture
def client(model):
    api = create_app(model)
    return testing.TestClient(api)


def test_predictions_route(client):
    img = Image.open("ebl_ai/tests/test_image.jpg")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    result = client.simulate_post(
        "/generate", body=byte_im, headers={"content-type": "image/png"}
    )
    assert result.status == falcon.HTTP_OK
    result_rectangle = result.json["boundaryResults"][0]
    assert isinstance(result_rectangle["top_left_x"], float)
