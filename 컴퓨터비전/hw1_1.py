import cv2
import numpy as np
import sys


def find_board_size(image):
    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive Thresholding을 적용하여 이미지를 이진화
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 체커 판의 가로줄과 세로줄을 찾기 위해 히스토그램을 계산
    horizontal_sum = np.sum(thresh, axis=1)
    vertical_sum = np.sum(thresh, axis=0)

    # 히스토그램을 기반으로 줄을 찾아내는 함수
    def find_lines(sum_array):
        line_count = 0
        threshold = thresh.shape[1] * 255 * 0.5  # 줄을 구분하기 위한 임계값 설정

        for i in range(len(sum_array)):
            if sum_array[i] < threshold:
                line_count += 1
                while i < len(sum_array) and sum_array[i] < threshold:
                    i += 1
        return line_count

    # 가로줄과 세로줄의 수 계산
    horizontal_lines = find_lines(horizontal_sum)
    vertical_lines = find_lines(vertical_sum)

    return horizontal_lines, vertical_lines


def main():
    # 명령행 인자로부터 이미지 파일 이름을 가져오기
    image_file = sys.argv[1]

    image = cv2.imread(image_file)
    if image is None:
        print("Image not found")
        return

    board_size = find_board_size(image)
    if board_size:
        print(f"{board_size[0]}X{board_size[1]}")


if __name__ == '__main__':
    main()
