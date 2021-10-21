from io import BytesIO

import attr
import falcon
import numpy as np
from PIL import Image

from ebl_ai.model import Model


class GenerateBoundingBoxesResource:
    def __init__(self, model: Model):
        self._model = model

    def on_post(self, req, resp: falcon.Response):
        image_bytes = req.bounded_stream.read()
        image = Image.open(BytesIO(image_bytes))
        image = np.asarray(image)
        bounding_boxes_predictions = self._model.predict(image)
        resp.media = {
            "boundaryResults": [
                attr.asdict(rectangle) for rectangle in bounding_boxes_predictions
            ]
        }
