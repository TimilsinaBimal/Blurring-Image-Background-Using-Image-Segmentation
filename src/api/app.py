from flask import Flask
from flask_restful import Api, Resource, reqparse
from preprocess import get_results

app = Flask(__name__)
api = Api(app)


# class Result(Resource):
#     def __init__(self):
#         self.reqparse = reqparse.RequestParser()
#         self.reqparse.add_argument("image", type=str, required=False)
#         super(Result, self).__init__()

#     def get(self):
#         predicted_mask, blurred_image = get_results()
#         args = self.reqparse.parse_args()
#         image = args["image"]

#         if image == "predicted_mask":
#             response = {
#                 "predicted_image": predicted_mask.numpy().tolist()
#             }

#         if image == "blurred_image":
#             response = {
#                 "blurred_image": blurred_image.numpy().tolist()
#             }

#         if not image:
#             response = {
#                 "predicted_image": predicted_mask.numpy().tolist(),
#                 "blurred_image": blurred_image.numpy().tolist()
#             }

#         return response


class URLS(Resource):
    def __init__(self):
        super(URLS, self).__init__()
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument("image", type=str)

    def get(self):
        original_image_path, predicted_image_path, blurred_image_path = get_results()
        args = self.reqparse.parse_args()
        image = args["image"]
        if image == "predicted_mask":
            response = {"predicted_mask": "predicted_mask.jpg"}

        if image == "blurred_image":
            response = {"blurred_image": "blurred_image.jpg"}

        if not image:
            response = {
                "predicted_mask": "predicted_mask.jpg",
                "blurred_image": "blurred_image.jpg"
            }
        return response


# api.add_resource(Result, "/api/result")
api.add_resource(URLS, "/api/result/urls")


if __name__ == "__main__":
    app.run(debug=True)
