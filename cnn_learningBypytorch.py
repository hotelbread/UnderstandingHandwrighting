#####################################
##똑같은 내용을 파이토치로 작업해 보기##
#####################################


import torch
import torchvision
from torchvision import models, transforms
import os
import sys
import numpy as np

# 토치버전 확인부터
print("Pytorch Version: ", torch.__version__)
print('Torchvision Version: ', torchvision.__version__)

#vgg-16 모델의 인스턴스 생성
## 인스턴스란 객체를 의미?
use_pretrained = True #학습된 파라미터 사용
net = models.vgg16(pretrained = use_pretrained)
net.eval()

# 1. 데이터셋 생성하기


# 2. 데이터셋 전처리

# 3. validation나누기

# 4. 모델 생성하기

# 5. 모델 학습,평가 및 저장

if __name__ == '__main__':
