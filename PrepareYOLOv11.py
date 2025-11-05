from ultralytics import YOLO

# โหลดโมเดล
model = YOLO("yolo11n.pt")

# รัน prediction
results = model.predict(source="https://ultralytics.com/images/zidane.jpg", save=True)

print("Detection complete!")