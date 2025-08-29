# dataTools/customDataloader.py

import glob
import numpy as np
import time
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from utilities.customUtils import *
from dataTools.dataNormalization import *
# 새로 만든 불량 화소 생성기를 가져옵니다.
from dataTools.badPixelGenerator import generate_bad_pixels
import os

class customDatasetReader(Dataset):
    def __init__(self, image_list, imagePathGT, height, width, transformation=True):
        self.image_list = image_list
        # gtPath와 targetPath가 동일하므로 imagePathGT는 사용하지 않아도 됩니다.
        self.imagePathGT = imagePathGT 
        self.imageH = height
        self.imageW = width
        normalize = transforms.Normalize(normMean, normStd)

        # 정답(GT) 이미지용 변환: 텐서화 및 정규화
        self.transformHRGT = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        # 입력(Input) 이미지용 변환: 텐서화 및 정규화
        self.transformRI = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    def __len__(self):
        return (len(self.image_list))
    
    def __getitem__(self, i):
        # ----------------------------------------------------------- #
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 로직 변경 시작 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ #
        # ----------------------------------------------------------- #
        
        # 1. 원본(GT) 이미지 불러오기
        # targetPath와 gtPath가 같으므로, image_list에서 바로 GT 이미지를 불러옵니다.
        try:    
            gt_image_pil = Image.open(self.image_list[i]).convert("RGB")
        except:
            # 파일 오류 시 다음 이미지로 건너뛰기
            return self.__getitem__(i + 1)

        # 2. GT 이미지를 Numpy 배열로 변환
        gt_image_np = np.array(gt_image_pil)
        
        # 3. 불량 화소 생성기를 사용하여 입력(Input) 이미지 생성
        input_image_np = generate_bad_pixels(gt_image_np)
        
        # 4. Numpy 배열을 다시 PIL 이미지로 변환
        input_image_pil = Image.fromarray(input_image_np)

        # 5. 각각의 이미지에 변환(transform) 적용
        self.inputImage = self.transformRI(input_image_pil)
        self.gtImageHR = self.transformHRGT(gt_image_pil)

        # ----------------------------------------------------------- #
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 로직 변경 종료 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ #
        # ----------------------------------------------------------- #

        return self.inputImage, self.gtImageHR