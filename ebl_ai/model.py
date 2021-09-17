import tempfile
from io import BytesIO
from typing import Union, List, Sequence

import attr
import numpy as np
from PIL import Image
from mmcv import Config
from mmocr.apis import init_detector, model_inference


@attr.s(auto_attribs=True, frozen=True)
class BoundingBoxesPrediction:
    top_left_x: float
    top_left_y: float
    width: float
    height: float
    probability: float


class Model:
    def __init__(self, configFile: str, checkpoint: str):
        self.model = init_detector(
            Config.fromfile(configFile), checkpoint=checkpoint, device="cpu"
        )

        if self.model.cfg.data.test["type"] == "ConcatDataset":
            self.model.cfg.data.test.pipeline = self.model.cfg.data.test["datasets"][
                0
            ].pipeline

    def _predict(self, image_path: str) -> List[List[float]]:
        """
        :param image_path: str
        :return: List[List[float]] of vertices of polygons with last
        entry beeing the score (probability between 0 - 1)
        """
        return model_inference(self.model, image_path)["boundary_result"]

    def _polygons_with_probabilites_to_rectangle(
        self, polygons_with_probabilites: Sequence[Sequence[float]]
    ) -> List[BoundingBoxesPrediction]:
        def _polygon_to_rectangle(
            polygon: Sequence[float], probability: float
        ) -> BoundingBoxesPrediction:
            x_coordinates = polygon[::2]
            y_coordinates = polygon[1::2]
            min_x, max_x = min(x_coordinates), max(x_coordinates)
            min_y, max_y = min(y_coordinates), max(y_coordinates)
            return BoundingBoxesPrediction(
                min_x, min_y, max_x - min_x, max_y - min_y, probability
            )

        rectangles = []
        for polygon in polygons_with_probabilites:
            rectangles.append(_polygon_to_rectangle(polygon[:-1], polygon[-1]))
        return rectangles

    def predict(
        self, image: Union[np.ndarray, str]
    ) -> Sequence[BoundingBoxesPrediction]:
        """
        Should be able to do inference on an image file and np.array
        Currently there is a bug for python images:
        KeyError: 'LoadImageFromNdarray is not in the pipeline registry'
        https://github.com/open-mmlab/mmdetection/issues
        that's why the image is saved to a file and then used for prediction
        """
        if isinstance(image, str):
            boundary_results = self._predict(image)
        else:
            image = Image.fromarray(image)
            with tempfile.NamedTemporaryFile(suffix="jpeg") as file:
                buf = BytesIO()
                image.save(buf, format="JPEG")
                file.write(buf.getvalue())
                boundary_results = self._predict(file.name)

        return self._polygons_with_probabilites_to_rectangle(boundary_results)
