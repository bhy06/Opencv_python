import cv2
import mediapipe as mp
import time


class FaceDetector:
    def __init__(self, min_detection_con=0.5):
        # Define min detection confidence
        self.min_detection_con = min_detection_con

        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(self.min_detection_con)

    def find_faces(self, img, draw=True):

        # Convert to RGB for face detection
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(img_rgb)
        bboxs = []

        if results.detections:
            # Multiple faces
            for id, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape

                # Convert unit coordinates based on image shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)

                bboxs.append([bbox, detection.score])
                if draw:
                    img = self.fancy_draw(img, bbox)

                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        return img, bboxs

    # Draw corners for bbox
    def fancy_draw(self, img, bbox, line_length=30, thickness=5, rec_thickness=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (255, 0, 255), rec_thickness)

        # Top Left x, y
        cv2.line(img, (x, y), (x + line_length, y), (255, 0, 255), thickness)
        cv2.line(img, (x, y), (x, y + line_length), (255, 0, 255), thickness)

        # Top Right x1, y
        cv2.line(img, (x1, y), (x1 - line_length, y), (255, 0, 255), thickness)
        cv2.line(img, (x1, y), (x1, y + line_length), (255, 0, 255), thickness)

        # Bottom Left x, y1
        cv2.line(img, (x, y1), (x + line_length, y1), (255, 0, 255), thickness)
        cv2.line(img, (x, y1), (x, y1 - line_length), (255, 0, 255), thickness)

        # Bottom Right x1, y1
        cv2.line(img, (x1, y1), (x1 - line_length, y1), (255, 0, 255), thickness)
        cv2.line(img, (x1, y1), (x1, y1 - line_length), (255, 0, 255), thickness)

        return img


def main():
    cap = cv2.VideoCapture(0)
    start_time = 0
    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img, bboxs = detector.find_faces(img)

        # Output fps
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        start_time = end_time
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
