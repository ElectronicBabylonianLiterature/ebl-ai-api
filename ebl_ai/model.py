import tempfile
from io import BytesIO
from typing import Union, List, Sequence

import attr
import numpy as np
from PIL import Image
from mmocr.apis import MMOCRInferencer

Image.MAX_IMAGE_PIXELS = None


@attr.s(auto_attribs=True, frozen=True)
class BoundingBoxesPrediction:
    top_left_x: float
    top_left_y: float
    width: float
    height: float
    probability: float


class Model:
    def __init__(self, configFile: str, checkpoint: str):
        self.model = MMOCRInferencer(det=configFile, det_weights=checkpoint)

    def _predict(self, image_path: str) -> List[List[float]]:
        x = self.model(image_path)["predictions"]
        result = []
        # _polygons_with_probabilites_to_rectangle expects [*polygon, probability]
        # this was default return value of the model but updating mmocr version changed it
        for polygons, scores in zip(x[0]["det_polygons"], x[0]["det_scores"]):
            result.append([*polygons, scores])
        return result

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
        if isinstance(image, str):
            boundary_results = self._predict(image)
        else:
            """
            Currently there is a bug for python images:
            KeyError: 'LoadImageFromNdarray is not in the pipeline registry'
            https://github.com/open-mmlab/mmdetection/issues
            that's why the image is saved to a file and then used for prediction
            """
            image = Image.fromarray(image)
            with tempfile.NamedTemporaryFile(suffix="jpeg") as file:
                buf = BytesIO()
                image.save(buf, format="JPEG")
                file.write(buf.getvalue())
                boundary_results = self._predict(file.name)

        return self._polygons_with_probabilites_to_rectangle(boundary_results)
