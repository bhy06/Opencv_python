import cv2
import mediapipe as mp
import time


class FaceMeshDetector:
    def __init__(self, mode=False, max_faces=1, min_detection_con=0.5,
                 min_tracking_con=0.5):
        self.mode = mode
        self.max_faces = max_faces  # Set max number of faces
        self.min_detection_con = min_detection_con
        self.min_tracking_con = min_tracking_con

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(self.mode, self.max_faces, self.min_detection_con,
                                                    self.min_tracking_con)
        # Set thickness and circle radius of mesh (468 points in total)
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=2)

    def find_face_mesh(self, img, draw=True):
        # Convert to RGB for face mesh
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        # List for faces
        faces = []

        if results.multi_face_landmarks:
            # Multiple faces
            for face_lms in results.multi_face_landmarks:
                # Draw face mesh
                if draw:
                    self.mp_draw.draw_landmarks(img, face_lms, self.mp_face_mesh.FACE_CONNECTIONS,
                                               self.draw_spec, self.draw_spec)
                face = []  # List for 468 points of face
                for id, lm in enumerate(face_lms.landmark):
                    ih, iw, ic = img.shape

                    # Convert unit coordinates based on image shape
                    x, y = int(lm.x * iw), int(lm.y * ih)

                    # Show id number
                    # cv2.putText(img, f'{id}', (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)

                    # print(id ,x, y)
                    face.append([x, y])
                faces.append(face)

        return img, faces


def main():
    cap = cv2.VideoCapture(0)
    start_time = 0
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        img, faces = detector.find_face_mesh(img)
        if len(faces) != 0:
            print(faces)

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        start_time = end_time
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
