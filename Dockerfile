# docker build -t detection . 
# docker run --rm detection
# Using python3.10 image as the base image
FROM python:3.10

# Setting the working directory inside the container
WORKDIR /app

# Installing system dependencies for OpenCV, PyTorch, and MiDaS
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Installing Python libraries
RUN pip install ultralytics opencv-python-headless numpy 
RUN pip install torch torchvision timm
RUN pip install google-cloud-pubsub apache-beam[gcp] 

# Downloading models if they dont exist
RUN if [ ! -f "/app/yolov8n.pt" ]; then \
        wget -O /app/yolov8n.pt "https://github.com/ultralytics/assets/releases/download/v8/yolov8n.pt"; \
    fi
RUN python -c "import torch; torch.hub.load('isl-org/MiDaS', 'MiDaS_small').eval()"

# # Copy dataset and scripts
# COPY Dataset_Occluded_Pedestrian /app/Dataset_Occluded_Pedestrian
# COPY Labels.csv /app/Labels.csv
# COPY detection.py /app/detection.py

# Run the detection script in the dataflow worker
ENTRYPOINT ["python", "detection.py", "--modelYolo", "/app/yolov8n.pt"]
