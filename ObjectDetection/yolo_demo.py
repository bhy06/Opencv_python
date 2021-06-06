import cv2
import numpy as np

conf_threshold = 0.5
nms_threshold = 0.2

cap = cv2.VideoCapture(0)
whT = 320

# Get 80 class names
class_names = []
class_file = "coco_yolo.names"
with open(class_file, "rt") as f:
    class_names = f.read().rstrip("\n").split("\n")

model_config = "yolov4-tiny.cfg"
model_weights = "yolov4-tiny.weights"

net = cv2.dnn.readNetFromDarknet(model_config, model_weights)

# Use CPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# # Use GPU
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


def find_objects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    class_ids = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)  # return index
            confidence = scores[class_id]
            if confidence > conf_threshold:

                # Convert unit coordination, yolo outputs center x, y, width and height
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int(det[0] * wT - w / 2), int(det[1] * hT - h / 2)

                bbox.append([x, y, w, h])
                class_ids.append(class_id)
                confs.append(float(confidence))

    # Performs non maximum suppression given boxes and corresponding scores
    indices = cv2.dnn.NMSBoxes(bbox, confs, conf_threshold, nms_threshold)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0:4]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f"{class_names[class_ids[i]].upper()}:{int(confs[i]*100)}%", (x, y-10),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)


while True:
    success, img = cap.read()

    # Preprocessing for inputs
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    # Get outputs
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)

    find_objects(outputs, img)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
