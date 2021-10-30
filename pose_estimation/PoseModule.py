import cv2
import mediapipe as mp
import time


class poseDetector():

    def __init__(self,
                 mode=False,
                 # upBody=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth=True,
                 detectionCon=0.5,
                 trackingCon=0.5):
        self.mode = mode
        # self.upBody = upBody
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model_complexity, self.smooth_landmarks, self.enable_segmentation,
                                     self.smooth, self.detectionCon, self.trackingCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm, in enumerate(self.results.pose_landmarks.landmark):
                h, w, c, = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList


def main():
    cap = cv2.VideoCapture('Videos/8.mp4')
    ok_flag = False
    if cap.isOpened():
        ok_flag = cap.isOpened()
    else:
        print("Cannot open camera")
        exit()
    pTime = 0
    detector = poseDetector()
    while ok_flag:
        success, img = cap.read()
        # if frame is read correctly ret is True
        if not success:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
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


if __name__ == "__main__":
    main()
