# utilities/inferenceUtils.py

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
import glob
from shutil import copyfile
import matplotlib.pyplot as plt
from utilities.customUtils import *
from dataTools.sampler import *
import numpy as np
import cv2
from PIL import Image
from dataTools.dataNormalization import *
from PIL import ImageFile
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 불량 화소 생성기 import ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
from dataTools.badPixelGenerator import generate_bad_pixels
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

ImageFile.LOAD_TRUNCATED_IMAGES = True

class AddGaussianNoise(object):
    def __init__(self, noiseLevel):
        self.var = 0.1
        self.mean = 0.0
        self.noiseLevel = noiseLevel

    def __call__(self, tensor):
        sigma = self.noiseLevel/100.
        noisyTensor = tensor + torch.randn(tensor.size()).uniform_(0, 1.) * sigma  + self.mean
        return noisyTensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.var)


class inference():
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ defectiveDir 파라미터 추가 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    def __init__(self, gridSize, inputRootDir, outputRootDir, modelName, resize = None, validation = None, defectiveDir = "./defective_output/"):
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        self.inputRootDir = inputRootDir
        self.outputRootDir = outputRootDir
        self.gridSize = gridSize
        self.modelName = modelName
        self.resize = resize
        self.validation = validation
        self.unNormalize = UnNormalize()
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 불량화소 이미지 저장 경로 설정 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        self.defectiveDir = defectiveDir
        createDir(self.defectiveDir)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲


    def inputForInference(self, imagePath, noiseLevel):
        img = Image.open(imagePath)

        if  ("McM" in imagePath) or ("WED" in imagePath) or ("BSD" in imagePath):
            resizeDimension =  (512, 512)
            img = img.resize(resizeDimension)
            img.save(imagePath)
        if "Urban" in imagePath:
            resizeDimension =  (1024, 1024)
            img = img.resize(resizeDimension)
            img.save(imagePath)

        img_np = np.asarray(img)
        if self.gridSize == 1 :
            img_np = bayerSampler(img_np)
        elif self.gridSize == 2 :
            img_np = quadBayerSampler(img_np)
        elif self.gridSize == 3 :
            img_np = dynamicBayerSampler(img_np, gridSze)
        
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 불량 화소 생성 및 저장 로직 추가 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # 1. 불량 화소 생성
        defective_img_np = generate_bad_pixels(img_np)

        # 2. 불량 화소가 적용된 이미지 저장
        datasetName = imagePath.split("/")[-2]
        defective_save_dir = os.path.join(self.defectiveDir, self.modelName, datasetName)
        createDir(defective_save_dir)
        defective_image_path = os.path.join(defective_save_dir, extractFileName(imagePath, True) + "_defective.png")
        Image.fromarray(defective_img_np).save(defective_image_path)

        # 3. PIL 이미지로 변환하여 이후 프로세스 진행
        img = Image.fromarray(defective_img_np)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        if self.resize:
            transform = transforms.Compose([ transforms.Resize(self.resize, interpolation=Image.BICUBIC) ])
            img = transform(img)

        transform = transforms.Compose([ transforms.ToTensor(),
                                        transforms.Normalize(normMean, normStd),
                                        AddGaussianNoise(noiseLevel=noiseLevel)])

        testImg = transform(img).unsqueeze(0)

        return testImg


    def saveModelOutput(self, modelOutput, inputImagePath, noiseLevel, step = None, ext = ".png"):
        datasetName = inputImagePath.split("/")[-2]
        if step:
            imageSavingPath = self.outputRootDir + self.modelName  + "/"  + datasetName + "/" + extractFileName(inputImagePath, True)  + \
                            "_sigma_" + str(noiseLevel) + "_" + self.modelName + "_" + str(step) + "_restored" + ext
        else:
            imageSavingPath = self.outputRootDir + self.modelName  + "/"  + datasetName + "/" + extractFileName(inputImagePath, True)  + \
                            "_sigma_" + str(noiseLevel) + "_" + self.modelName + "_restored" + ext

        save_image(self.unNormalize(modelOutput[0]), imageSavingPath)



    def testingSetProcessor(self):

        testSets = glob.glob(self.inputRootDir+"*/")
        if self.validation:
            testSets = testSets[:1]
        testImageList = []
        for t in testSets:
            testSetName = t.split("/")[-2]
            createDir(self.outputRootDir + self.modelName  + "/" + testSetName)
            imgInTargetDir = imageList(t, False)
            testImageList += imgInTargetDir
        return testImageList
