import cv2
import mediapipe as md
import time
import HandTrackingModule as htm

pTime = 0
cTime = 0
cap = cv2.VideoCapture(1)
detector = htm.handDetector()

ok_flag = False

if cap.isOpened():
    ok_flag = cap.isOpened()
else:
    print("Cannot open camera")
    exit()

while ok_flag:
    success, img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        print(lmList[4])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3, )

    cv2.imshow("Image", img)
    # cv2.waitKey(1)
    if cv2.waitKey(1) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        cap.release()
        cv2.destroyAllWindows()
        break
