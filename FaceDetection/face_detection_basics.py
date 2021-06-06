import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
start_time = 0

mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(0.75)

while True:
    success, img = cap.read()

    # Convert to RGB for face detection
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if results.detections:
        # Multiple faces
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape

            # Convert unit coordinates based on image shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'Pred: {int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    start_time = end_time
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
