import cv2
import numpy as np

conf_threshold = 0.5  # Threshold to detect objects
nms_threshold = 0.2  # NMS threshold to detect objects, less value suppresses more

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height

# Get 91 class names
class_names = []
class_file = "coco_ssd.names"
with open(class_file, "rt") as f:
    class_names = f.read().rstrip("\n").split("\n")

config_path = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weights_path = "frozen_inference_graph.pb"

# Settings
net = cv2.dnn_DetectionModel(weights_path, config_path)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    class_ids, confs, bbox = net.detect(img, confThreshold=conf_threshold)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    # Performs non maximum suppression given boxes and corresponding scores
    indices = cv2.dnn.NMSBoxes(bbox, confs, conf_threshold, nms_threshold)  # list
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0:4]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f"{class_names[class_ids[i][0]-1].upper()}:{int(confs[i]*100)}%",
                    (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
