import os
from google.cloud import pubsub_v1      #pip install google-cloud-pubsub
import glob

# Search the current directory for the JSON file (including the service account key) 
# to set the GOOGLE_APPLICATION_CREDENTIALS environment variable.
files=glob.glob("*.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=files[0];

project_id = "directed-post-449003-u2"
subscription_id = "box_and_depth-sub"

subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, subscription_id)

import json
def callback(message: pubsub_v1.subscriber.message.Message) -> None:
    #print(f"Received {json.loads(message)}.")
    msgData = json.loads(message.data.decode("utf-8"))
    jsonData = json.loads(msgData)
    print(f"Received {jsonData}.")
    message.ack()

streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
print(f"Listening for messages on {subscription_path}..\n")

with subscriber:
    streaming_pull_future.result()




