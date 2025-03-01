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
topic_name = "car_view_image"; 
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_name)

df=pd.read_csv('Labels.csv')

# going through every image
for _, row in df.iterrows():
    image = row["Occluded_Image_view"]

    print(image)

    # creating the image path
    imagepath = os.path.join(IMAGE_PATH, image)

    if os.path.exists(imagepath):

        # opening the image and publishing it
        try:
            # reading the image and coverting it to hex
            with open(imagepath, "rb") as f:
                image_bytes = f.read()
                image_hex = image_bytes.hex()

            message = json.dumps({"image": image_hex, "id": image}).encode("utf-8")


            future = publisher.publish(topic_path, message);
            #ensure that the publishing has been completed successfully
            future.result()    
            print("The messages has been published successfully")
        except: 
            print("Failed to publish the message")
    else:
        print("No image found.")



           
