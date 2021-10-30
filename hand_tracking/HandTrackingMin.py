import cv2
import mediapipe as mp
import time
import sys

# noinspection PyUnresolvedReferences
camera_port = 1
cap = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
ok_flag = False

if cap.isOpened():
    ok_flag = cap.isOpened()
else:
    print("Cannot open camera")
    exit()



while ok_flag:

    success, img = cap.read()

    # if frame is read correctly ret is True
    if not success:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # noinspection PyUnresolvedReferences
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c, = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id == 0:
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)


            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # noinspection PyUnresolvedReferences
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3, )

    # noinspection PyUnresolvedReferences
    cv2.imshow("Image", img)

    if cv2.waitKey(1) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        cap.release()
        cv2.destroyAllWindows()
        break


# When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()