import cv2
import time
import numpy as np
import HandTracking as ht
import math

from helper import load_variables
from Colors import Color


def window_is_closed(window_name) -> bool:
    # If the q key is pressed, terminate the while loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        return True

    # If the Esc key is pressed, terminate the while loop
    if cv2.waitKey(1) == 27:
        return True

    # If X in the top right corner is pressed, terminate the while loop
    if not cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE):
        return True

    return False


def draw(img, volBar, volPer, volume, center_frame_w, colorVol) -> None:
    # Drawings
    cv2.rectangle(img, (50, 150), (85, 400), Color.BLUE, 1)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), Color.BLUE, cv2.FILLED)
    cv2.putText(
        img,
        f"{int(volPer)} %",
        (40, 450),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        Color.BLUE,
        1,
    )

    cVol = volume.GetMasterVolumeLevelScalar() * 100
    cVol = math.ceil(round(cVol, 2))
    cv2.putText(
        img,
        f"Vol Set: {int(cVol)}",
        (center_frame_w, 50),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        colorVol,
        1,
    )

    # volPer = cVol


def compute_fps(pTime, img):
    # Frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(
        img,
        f"FPS: {int(fps)}",
        (40, 50),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        Color.BLUE,
        1,
    )

    return pTime, img


def process_data(area, img, fingers_dict, detector, volume):
    if 250 < area:

        # Find distance between two fingers
        finger1 = fingers_dict.get("index")
        finger2 = fingers_dict.get("thumb")

        length, img, lineInfo = detector.find_distance(finger1, finger2, img)
        # print(length)

        # data_finger1 = lmlist[finger1]
        # data_finger2 = lmlist[finger2]
        # print(f"\n{data_finger1 = }", f"{data_finger2 = }")

        # Convert volume
        volBar = np.interp(length, [50, 200], [400, 150])
        volPer = np.interp(length, [50, 200], [0, 100])

        # Reduce resolution to make it smoother
        smoothness = 10
        volPer = smoothness * round(volPer / smoothness)

        # Check fingers up
        fingers = detector.fingers_up()

        # If pinky finger is down, then set volume
        if not fingers[4]:
            volume.SetMasterVolumeLevelScalar(volPer / 100, None)
            # cv2.circle(
            #    img, (lineInfo[4], lineInfo[5]), 15, Color.GREEN, cv2.FILLED
            # )
            colorVol = Color.GREEN
        else:
            colorVol = Color.BLUE

    return img, volBar, volPer, volume, colorVol


def main():
    index_cam, width_cam, height_cam, flipCode, window_name = load_variables(
        name="camera"
    )

    cap = cv2.VideoCapture(index=index_cam)

    if not cap.isOpened():
        print("Camera could not be open")
        exit()

    cap.set(3, width_cam)
    cap.set(4, height_cam)

    frame_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    center_frame_w = int(frame_w / 2)

    detector = ht.HandDetector(min_detection_confidence=0.7, max_num_hands=4)

    volume, volPer, volBar = load_variables(name="volume")

    area, fingers_dict = load_variables(name="fingers")

    colorVol = Color.BLUE
    pTime = 0  # Used in compute_fps()

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Can't receive frame. Exiting ...")
            break

        img = cv2.flip(img, flipCode=flipCode)
        img = detector.find_hands(img)
        lmlist, bbox = detector.find_position(img, draw=True)

        if not lmlist:
            draw(img, volBar, volPer, volume, center_frame_w, colorVol)

            pTime, img = compute_fps(pTime, img)

            cv2.imshow(window_name, img)

            # Check is user decided to close the window
            if window_is_closed(window_name):
                break

            continue

        # Filter based on size
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
        # print(f"{area = }")

        img, volBar, volPer, volume, colorVol = process_data(
            area, img, fingers_dict, detector, volume
        )

        draw(img, volBar, volPer, volume, center_frame_w, colorVol)

        pTime, img = compute_fps(pTime, img)

        cv2.imshow(window_name, img)

        # Check is user decided to close the window
        if window_is_closed(window_name):
            break

    # When everything done, release the capture
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
