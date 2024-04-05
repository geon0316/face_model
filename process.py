from keras.models import load_model
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import keras
from keras.utils import to_categorical
import cv2
from scipy.ndimage import zoom
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

import utilFunc



# 불러올 모델 경로
model_path = 'faceModel.keras'

# 모델 불러오기
model_ = load_model(model_path)

while(True):

    print("while 문 진입")

    image_path = 'no.jpg'
    face = cv2.imread(image_path)

    print("얼굴 데이터",face)

    # 얼굴 추출
    gray, detected_faces, coord = utilFunc.detect_face(face)
    face_zoom = utilFunc.extract_face_features(gray, detected_faces, coord)

    # 모델 추론
    input_data = np.reshape(face_zoom[0].flatten(), (1, 48, 48, 1))
    output_data = model_.predict(input_data)
    result = np.argmax(output_data)

    # 결과 문자로 변환
    if result == 0:
        emotion = 'angry'
    elif result == 1:
        emotion = 'disgust'
    elif result == 2:
        emotion = 'fear'
    elif result == 3:
        emotion = 'happy'
    elif result == 4:
        emotion = 'sad'
    elif result == 5:
        emotion = 'surprise'
    elif result == 6:
        emotion = 'neutral'

    print("감정상태: ", emotion)
    print(result)

    # 시각화
    plt.subplot(121)
    plt.title("Original Face")
    plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

    plt.subplot(122)
    plt.title(f"Extracted Face : {emotion}")
    plt.imshow(face_zoom[0])
    break

print("끝")