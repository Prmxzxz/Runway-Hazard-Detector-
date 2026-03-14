from ultralytics import YOLO

# Load a model
model = YOLO("yolo26m.pt")

# Train the model
train_results = model.train(
    data=r"Runway-Hazard-Detector-1/data.yaml",
    epochs=50,
    imgsz=640,
    device=0,   # your GPU is CUDA:0
    workers=0
)

# Evaluate model performance on the validation set
metrics = model.val()