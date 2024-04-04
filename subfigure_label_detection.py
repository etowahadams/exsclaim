from ultralytics import YOLO

def detect_subfigure_labels(img_path):
    model = YOLO("weights/yolov8-subfigure-label.pt")
    results = model.predict(img_path, conf=0.1)
    bboxes = results[0].boxes.xyxy.cpu().numpy().astype(int).tolist()
    conf = results[0].boxes.conf.cpu().numpy().tolist()
    bboxes = [bbox + [c] for bbox, c in zip(bboxes, conf)]
    return bboxes

if __name__ == "__main__":
    img_path = 'bmc2.jpeg'
    bbox = detect_subfigure_labels(img_path)
    print(bbox)
