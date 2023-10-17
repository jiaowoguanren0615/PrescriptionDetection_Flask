from functions import GetPicTure, get_prediction
from flask import Flask, jsonify, request
from flask_cors import CORS


app = Flask(__name__)

CORS(app)


@app.route("/predict_coordinate", methods=["POST", 'GET'])
def predict():
    image_url = request.args.get('image_url')
    image_path = GetPicTure(image_url)
    info = get_prediction(val_image_root_path=image_path)
    return jsonify(info)

if __name__ == '__main__':

    app.run(host="0.0.0.0", port=15003, debug=True)