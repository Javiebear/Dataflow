# environment variable setup for private key file
import os
import pandas as pd   #pip install pandas  ##to install
from google.cloud import pubsub_v1    #pip install google-cloud-pubsub  ##to install
import glob
import base64
import os 
import json

IMAGE_PATH = "Dataset_Occluded_Pedestrian"

# Searching for the json file and setting it to the google credentials
files=glob.glob("*.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=files[0];

# Setting all the correct IDs
project_id="directed-post-449003-u2";

# publisher for the topic adhearing to the car view data images of pedestrians
topic_name = "car_image_view"; 
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_name)

df=pd.read_csv('Labels.csv')

for row in df:
    image = row["Occluded_Image_view"]
    # opening the image and publishing it
    try:
        with open(f"{IMAGE_PATH}/{image}", "rb") as f:
            value =  base64.b64encode(f.read());   # read the image and serizalize it to base64

        future = publisher.publish(topic_path, value);
        #ensure that the publishing has been completed successfully
        future.result()    
        print("The messages has been published successfully")
    except: 
        print("Failed to publish the message")



           
