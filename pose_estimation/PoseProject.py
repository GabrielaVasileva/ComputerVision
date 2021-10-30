import cv2
import time
import PoseModule as pm

cap = cv2.VideoCapture('Videos/14.mp4')
ok_flag = False
if cap.isOpened():
    ok_flag = cap.isOpened()
else:
    print("Cannot open camera")
    exit()
pTime = 0
detector = pm.poseDetector()
while ok_flag:
    success, img = cap.read()
    # if frame is read correctly ret is True
    if not success:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    if len(lmList)!=0:
        print(lmList[14])
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 10, (0, 0, 100), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)

    cv2.waitKey(10)

    if cv2.waitKey(10) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        cap.release()
        cv2.destroyAllWindows()
        break
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)

    cv2.waitKey(10)

    if cv2.waitKey(1) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        cap.release()
        cv2.destroyAllWindows()
        break
