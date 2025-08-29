# dataTools/badPixelGenerator.py

import numpy as np
import random

def add_single_bad_pixels(image_np, percentage=0.01):
    """이미지에 단일 불량 화소를 추가합니다."""
    img = image_np.copy()
    h, w, c = img.shape
    num_bad_pixels = int(h * w * percentage)

    coords_y = np.random.randint(0, h, size=num_bad_pixels)
    coords_x = np.random.randint(0, w, size=num_bad_pixels)

    for y, x in zip(coords_y, coords_x):
        img[y, x, :] = 0 if random.random() < 0.5 else 255
        
    return img

# ========================================================== #
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 이 함수를 수정합니다 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ #
# ========================================================== #
def add_cluster_bad_pixels(image_np, percentage=0.005):
    """
    쿼드 베이어 패턴에 맞춰 2x2 크기의 클러스터 불량 화소를 추가합니다.
    클러스터는 항상 짝수 좌표(0, 2, 4...)에서 시작합니다.
    """
    img = image_np.copy()
    h, w, c = img.shape
    num_bad_pixels_total = int(h * w * percentage)
    num_bad_pixels_generated = 0
    
    # 클러스터 크기는 2x2로 고정
    cluster_size = 2 * 2

    # 이미지 높이와 너비가 2보다 작으면 실행하지 않음
    if h < 2 or w < 2:
        return img

    while num_bad_pixels_generated < num_bad_pixels_total:
        # 클러스터 시작 위치는 항상 (짝수, 짝수) 좌표가 되도록 랜덤 선택
        # (h - 2) // 2 는 2x2 블록이 들어갈 수 있는 y축 시작점의 최대 인덱스
        start_y = random.randint(0, (h - 2) // 2) * 2
        start_x = random.randint(0, (w - 2) // 2) * 2

        # 2x2 클러스터 영역을 0 (검은색)으로 마스킹
        img[start_y : start_y + 2, start_x : start_x + 2, :] = 0
        num_bad_pixels_generated += cluster_size
        
    return img
# ========================================================== #
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 수정 완료 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ #
# ========================================================== #


def add_column_bad_pixels(image_np, max_bad_columns=2):
    """이미지에 세로줄 형태의 불량 화소를 추가합니다."""
    img = image_np.copy()
    h, w, c = img.shape
    num_bad_columns = random.randint(1, max_bad_columns)
    
    bad_column_indices = random.sample(range(w), num_bad_columns)

    for col_idx in bad_column_indices:
        img[:, col_idx, :] = 0
        
    return img

def generate_bad_pixels(image_np):
    """
    위 함수들을 모두 호출하여 이미지에 복합적인 불량 화소를 생성합니다.
    """
    corrupted_img = image_np.copy()
    
    corrupted_img = add_single_bad_pixels(corrupted_img, percentage=0.01)
    # 수정된 클러스터 함수를 호출합니다.
    corrupted_img = add_cluster_bad_pixels(corrupted_img, percentage=0.005)
    corrupted_img = add_column_bad_pixels(corrupted_img, max_bad_columns=2)
    
    return corrupted_img