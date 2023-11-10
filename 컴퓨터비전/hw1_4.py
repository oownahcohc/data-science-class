import cv2
import numpy as np
import sys


def find_checkerboard_corners(image):
    # 체커보드 모서리를 찾아 정렬
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour

    epsilon = 0.1 * cv2.arcLength(largest_contour, True)
    corners = cv2.approxPolyDP(largest_contour, epsilon, True)

    return corners.reshape(4, 2)


def perspective_transform(image, corners):
    # 투시 변환을 적용하여 이미지를 정방형으로 만들고,정렬된 모서리를 사용하여 매핑
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]

    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped


def filter_colors(hsv_image, lower_hsv, upper_hsv):
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    return mask


def count_pieces(image, lower_hsv, upper_hsv):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = filter_colors(hsv, lower_hsv, upper_hsv)

    # Contours 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 적절한 크기의 객체만을 세기
    piece_count = 0
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # 최소 크기 설정
            piece_count += 1

    return piece_count


def main():
    image_file = sys.argv[1]

    original_image = cv2.imread(image_file)

    corners = find_checkerboard_corners(original_image)
    transformed_image = perspective_transform(original_image, corners)

    bright_lower_hsv = np.array([20, 100, 150])
    bright_upper_hsv = np.array([30, 255, 255])
    dark_lower_hsv = np.array([0, 0, 0])
    dark_upper_hsv = np.array([180, 255, 50])

    bright_piece_count = count_pieces(transformed_image, bright_lower_hsv, bright_upper_hsv)
    dark_piece_count = count_pieces(transformed_image, dark_lower_hsv, dark_upper_hsv)

    print(f'w: {bright_piece_count} b: {dark_piece_count}')


if __name__ == '__main__':
    main()
