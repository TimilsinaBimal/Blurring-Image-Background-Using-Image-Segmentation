from flask import Flask
from flask_restful import Api, Resource
from api.preprocess import get_image_url, prepare_image, predict, blur_image, BASE_URL
from PIL import Image

app = Flask(__name__)
api = Api(app)


def save_img(img, filepath):
    img = Image.fromarray(img)
    img.save(filepath)


def get_results(original=False, predicted=False, blurred=False, save=False):
    image_url = get_image_url(BASE_URL)
    original_image = prepare_image(image_url)
    if original:
        return original_image
    predicted_mask = predict(original_image)
    if predicted:
        return predicted_mask
    blurred_image = blur_image(original_image, predicted_mask)
    if blurred:
        return blurred_image
    if save:
        save_img(original_image, "results/original_image.jpg")
        save_img(predicted_mask, "results/predicted_mask.jpg")
        save_img(blurred_image, "results/blurred_image.jpg")
        return

    return original_image, predicted_mask, blurred_image


class OriginalImage(Resource):
    def get(self):
        original_image = get_results(original=True)
        return {"original_image": original_image.numpy().tolist()}


class PredictedMask(Resource):
    def get(self):
        predicted_mask = get_results(predicted=True)
        return {"predicted_mask": predicted_mask.numpy().tolist()}


class BlurredImage(Resource):
    def get(self):
        blurred_image = get_results(blurred=True)
        return {"blurred_image": blurred_image.numpy().tolist()}


class Result(Resource):
    def get(self):
        original_image, predicted_mask, blurred_image = get_results()
        response = {
            "original_image": original_image.numpy().tolist(),
            "predicted_image": predicted_mask.numpy().tolist(),
            "blurred_image": blurred_image.numpy().tolist()
        }
        return response


class URLS(Resource):
    def get(self):
        get_results(save=True)
        response = {
            "orginal_image": "original_image.jpg",
            "predicted_mask": "predicted_mask.jpg",
            "blurred_image": "blurred_image.jpg"
        }
        return response


api.add_resource(Result, "/result/")
api.add_resource(OriginalImage, "/result/original/")
api.add_resource(PredictedMask, "/result/predicted/")
api.add_resource(BlurredImage, "/result/blurred/")
api.add_resource(URLS, "/result/urls/")


if __name__ == "__main__":
    app.run(debug=True)
