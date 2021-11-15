import falcon

from ebl_ai.generate_bounding_boxes_resource import GenerateBoundingBoxesResource
from ebl_ai.model import Model


def create_app(model: Model):
    app = falcon.App()
    generate_bounding_boxes = GenerateBoundingBoxesResource(model)
    app.add_route("/generate", generate_bounding_boxes)
    return app


def get_app():
    model = Model(
        configFile="model/fcenet_no_dcvn.py",
        checkpoint="model/checkpoint1.pth",
    )
    return create_app(model)
