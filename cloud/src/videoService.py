import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands


def finger_vector(lm, tip, base):
    tip = np.array([lm[tip].x, lm[tip].y])
    base = np.array([lm[base].x, lm[base].y])
    return tip - base


def normalize(v):
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def parallel(v1, v2, thresh=0.9):
    v1 = normalize(v1)
    v2 = normalize(v2)
    return abs(np.dot(v1, v2)) > thresh


def fingers_parallel(lm):
    v1 = finger_vector(lm, 8, 5)
    v2 = finger_vector(lm, 12, 9)
    v3 = finger_vector(lm, 16, 13)
    v4 = finger_vector(lm, 20, 17)

    return (
        parallel(v1, v2) and
        parallel(v1, v3) and
        parallel(v1, v4)
    )


def finger_up(lm, tip, pip, mcp):
    return lm[tip].y < lm[pip].y and lm[pip].y < lm[mcp].y


def count_fingers(lm):

    index_up = finger_up(lm, 8, 6, 5)
    middle_up = finger_up(lm, 12, 10, 9)
    ring_up = finger_up(lm, 16, 14, 13)
    pinky_up = finger_up(lm, 20, 18, 17)

    return sum([index_up, middle_up, ring_up, pinky_up])


def palm_normal(lm):
    A = np.array([lm[0].x, lm[0].y])
    B = np.array([lm[5].x, lm[5].y])
    C = np.array([lm[17].x, lm[17].y])

    AB = B - A
    AC = C - A

    normal = np.array([
        AB[0]*AC[1] - AB[1]*AC[0]
    ])

    return normal


def palm_facing_camera(lm):

    A = np.array([lm[0].x, lm[0].y])
    B = np.array([lm[5].x, lm[5].y])
    C = np.array([lm[17].x, lm[17].y])

    AB = B - A
    AC = C - A

    cross = AB[0]*AC[1] - AB[1]*AC[0]

    return cross < 0


def finger_direction(lm):

    v = finger_vector(lm, 12, 9)

    dx, dy = v

    if abs(dy) > abs(dx):
        if dy < 0:
            return "up"
    else:
        if dx < 0:
            return "left"
        else:
            return "right"

    return "unknown"


def gesture_from_image(path):

    image = cv2.imread(path)

    if image is None:
        return 0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7
    ) as hands:

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if not result.multi_hand_landmarks:
            return 0

        lm = result.multi_hand_landmarks[0].landmark

        finger_num = count_fingers(lm)

        if finger_num < 4:
            return 0

        if not fingers_parallel(lm):
            return 0

        direction = finger_direction(lm)

        palm_front = palm_facing_camera(lm)

        if direction == "up" and palm_front:
            return 1

        if direction == "left":
            return 2

        if direction == "right":
            return 3

        return 0


if __name__ == "__main__":

    img = "D:\\Doc\\biYeSheJi\\cloud\\src\\hand.png"

    gesture = gesture_from_image(img)

    print("gesture =", gesture)