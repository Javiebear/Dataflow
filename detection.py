import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os
from google.cloud import pubsub_v1      #pip install google-cloud-pubsub
import glob
import argparse
import json
import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

def singleton(cls):
    instances = {}
    def getinstance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return getinstance

# setting up the yolo model for bounding boxes
@singleton
class ModelYOLO:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 

    def predict(self, image):
        return self.model(image)

# setting up the ModelMiDas for depth detetion
@singleton
class ModelMiDas:
    def __init__(self):
        self.model = torch.hub.load("isl-org/MiDaS", "MiDaS_small")
        self.model.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.eval()
        self.transform = torch.hub.load("isl-org/MiDaS", "transforms").small_transform

    def predict(self, image):
        image = self.transform(image).to(next(self.model.parameters()).device)
        with torch.no_grad():
            prediction = self.model(image)
        return prediction.squeeze().cpu().numpy()

# running the code to receive a message of an image from the pub/sub, getting 
class getBoxAndDepth(beam.DoFn):

    # model paths
    def __init__(self, modelYoloPath):
        self.modelYoloPath = modelYoloPath

    def setup(self):
        self.modelYolo = ModelYOLO(self.modelYoloPath)
        self.modelMiDas = ModelMiDas()

    def process(self, element):
        # Converting the read image into an image
        message = json.loads(element.decode("utf-8"))  
        imageBytes = message["image"]
        imageID = message["id"]

        # Decoding image
        nparr = np.frombuffer(bytes.fromhex(imageBytes), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Predict the images boudning box of pedestrian and the depth 
        results = self.modelYolo.predict(image)  # Fixed
        depth_map = self.modelMiDas.predict(image)  # Fixed

        pedestrians = []
        # This for loop loops through each detected person in the image
        for result in results:
            for box in result.boxes:
                if int(box.cls) == 0: 
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    depth_roi = depth_map[y1:y2, x1:x2]
                    avg_depth = np.mean(depth_roi)
                    scaling_factor = 39.545  # Convert depth map values to real-world distances
                    distance = avg_depth / scaling_factor

                    pedestrians.append({
                        "bounding_box": [x1, y1, x2, y2],
                        "distance": round(distance, 2)
                    })

        # Format output message
        output_message = json.dumps({"id": imageID, "pedestrians": pedestrians})
        yield output_message



def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', required=True, help='Input file to process.')
    parser.add_argument('--output', dest='output', required=True, help='Output file to write results to.')
    parser.add_argument('--modelYolo', required=True, help='YOLO model path')
    parser.add_argument('--modelMiDas', required=True, help='MiDaS model path')

    known_args, pipeline_args = parser.parse_known_args(argv)
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True;
    with beam.Pipeline(options=pipeline_options) as p:
        (
            p
            | "ReadFromPubSub" >> beam.io.ReadFromPubSub(subscription=known_args.input)
            | "DetectPedestrians" >> beam.ParDo(getBoxAndDepth(known_args.modelYolo))
            | "WriteToPubSub" >> beam.io.WriteToPubSub(topic=known_args.output)
        )

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()
