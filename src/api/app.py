from flask import Flask
from flask_restful import Api, Resource
from api.preprocess import get_image_data, prepare_image, predict, blur_image, BASE_URL
import matplotlib.pyplot as plt

app = Flask(__name__)
api = Api(app)


class BlurImage(Resource):
    def get(self):
        # original_image = get_image_data(BASE_URL)
        original_image = prepare_image("src/test/test.JPG")
        predicted_mask = predict(original_image)
        plt.imshow(predicted_mask)
        blurred_image = blur_image(original_image, predicted_mask)
        return {"original_image": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "predicted_image": predicted_mask.numpy().tolist()}


api.add_resource(BlurImage, "/result/")

# if __name__ == "__main__":
#     app.run(debug=True)
