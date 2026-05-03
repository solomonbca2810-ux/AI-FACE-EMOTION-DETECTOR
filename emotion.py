from facial_emotion_recognition import EmotionRecognition
import cv2
er = EmotionRecognition(device='cpu')
cam = cv2.VideoCapture(0)
while True:
    success, frame = cam.read()
    if not success or frame is None:
        print("Camera not working ❌")
        continue
    frame = er.recognise_emotion(frame, return_type='BGR')
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break
cam.release()
cv2.destroyAllWindows()
