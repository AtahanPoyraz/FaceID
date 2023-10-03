import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            
            landmark_points = []
            
            for n in range(68):  
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmark_points.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

            landmark_points = np.array(landmark_points)

            print(landmark_points)

        cv2.imshow("Yüz Tanıma", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
