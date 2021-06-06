import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
start_time = 0

mp_draw = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
# Set thickness and circle radius of mesh (468 points in total)
draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=2)

while True:
    success, img = cap.read()

    # Convert to RGB for face mesh
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        # Multiple faces
        for face_lms in results.multi_face_landmarks:
            # Draw face mesh
            mp_draw.draw_landmarks(img, face_lms, mp_face_mesh.FACE_CONNECTIONS, draw_spec, draw_spec)

            for id, lm in enumerate(face_lms.landmark):
                ih, iw, ic = img.shape

                # Convert unit coordinates based on image shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                # print(id ,x, y)

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    start_time = end_time
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
