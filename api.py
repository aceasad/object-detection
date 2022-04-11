from Detector import YOLOV5_Detector

from flask import Flask, jsonify, request,render_template
from flask_cors import CORS
import os

import boto3
from dotenv import load_dotenv,find_dotenv
from botocore.exceptions import ClientError

load_dotenv(find_dotenv())
static_dir = str(os.path.abspath(os.path.join(__file__ , "..", 'templates/')))

app = Flask(__name__, static_folder=static_dir, static_url_path="", template_folder=static_dir)
cors = CORS(app, resources={r"/api/*": {"origins": '*'}})

s3_client = boto3.client(
's3',
aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
)
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
print(S3_BUCKET)
download_path = "downloadModel/best.pt"
download_file = "best.pt"
disable_print = True


basepath_video = 'apiData/video/'
basepath_json = 'apiData/json/'

@app.route('/api/downloadModel',methods=['GET'])
def download_model():
    if os.path.isfile(download_file):
        os.remove("best.pt")
        if disable_print:
            print ("Existing Model was removed")
    s3_client.download_file(S3_BUCKET, download_path, download_file)
    if disable_print:
        print("Model Downloaded Successfully!")
    return jsonify({"message":"New model has been downloaded for inference"})

@app.route('/',methods=['GET'])
def index():
    return render_template("index.html")


def saveFileLocal(data,name):
    FILE_OUTPUT = basepath_video+name

    # Checks and deletes the output file
    # You cant have a existing file or it will through an error
    if os.path.isfile(FILE_OUTPUT):
        os.remove(FILE_OUTPUT)

    # opens the file 'output.avi' which is accessable as 'out_file'
    with open(FILE_OUTPUT, "wb") as out_file:  # open for [w]riting as [b]inary
        out_file.write(data)
        return basepath_video+name

@app.route('/api/generateJson', methods=['POST'])
def predictor():
    data = request.files['videoFile'].read()
    name = request.files['videoFile'].filename
    saveFileLocal(data,name)
    detector = YOLOV5_Detector(weights='best.pt',
                            img_size=640,
                            confidence_thres=0.25,
                            iou_thresh=0.45,
                            agnostic_nms=True,
                            augment=True)
    detector.detect_on_video(name)

    return jsonify({"message":"Please download the model via /download endpoint first to create inference!"})


if __name__ == '__main__':
    app.run(port=8080, debug=True, host='127.0.0.1')
