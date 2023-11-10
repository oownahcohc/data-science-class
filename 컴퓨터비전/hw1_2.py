import cv2
import numpy as np
import sys

# 전역 변수
points = []  # 클릭된 점을 저장할 리스트


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        mark_point(image, (x, y))
        if len(points) == 4:
            perform_perspective_transform(points)


def mark_point(image, point):
    cv2.circle(image, point, 5, (255, 0, 0), -1)
    cv2.imshow('image', image)


def perform_perspective_transform(pts):
    pts1 = np.float32(pts)
    side = calculate_max_side_length(pts1)
    pts2 = np.float32([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(image, matrix, (int(side), int(side)))
    show_transformed_image(result)


def calculate_max_side_length(pts):
    return max([np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)])


def show_transformed_image(image):
    cv2.imshow('Transformed Image', image)


def main():
    global image
    image_file = sys.argv[1]

    image = cv2.imread(image_file)
    if image is None:
        print("Image not found")
        return

    cv2.imshow('image', image)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
