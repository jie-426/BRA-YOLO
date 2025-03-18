from ultralytics import YOLOv10
model = YOLOv10(r"E:\desktop\yolov10-main\runs\detect\train18\weights\best.pt")
results = model.predict(source=r'E:\desktop\2222.avi',save=True,save_frames=True)