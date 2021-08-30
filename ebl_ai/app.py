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
        configFile="../model/cpu/fcenet.py",
        checkpoint="../model/cpu/fcenet_r50_fpn_1500e_icdar2015-d435c061.pth",
    )
    return create_app(model)
