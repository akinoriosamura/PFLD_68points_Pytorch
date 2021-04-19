from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import cv2
import os

from tensorflow.lite.python.interpreter import Interpreter
from RetinaFaceMaster.test import predict


def inference(image):
    model_file = 'models2/wflw_moruhard_grow_68_Res50/wflw_moruhard_grow_68_Res50.tflite'
    # model_file = 'models2/wflw_moruhard_grow_68_Res50/all_model_85_opt.tflite'
    height, width, _ = image.shape
    boxes, _ = predict(image)

    # インタプリタの生成
    interpreter = Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    # 入力情報と出力情報の取得
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 幅と高さの取得
    # import pdb; pdb.set_trace()
    image_size  = input_details[0]['shape'][1]
    # image_size = input_details[0]['shape'][2]

    # import pdb; pdb.set_trace()
    for box in boxes:
        x1, y1, x2, y2 = (box[:4]+0.5).astype(np.int32)
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        size = int(max([w, h])*1.1)
        cx = x1 + w//2
        cy = y1 + h//2
        x1 = cx - size//2
        x2 = x1 + size
        y1 = cy - size//2
        y2 = y1 + size

        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)
        cropped = image[y1:y2, x1:x2]

        #print(dy, edy, dx, edx)
        if dx > 0 or dy > 0 or edx > 0 or edy > 0:
            cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)  # top,bottom,left,right
        cv2.imwrite('samples/croped.jpg', cropped)
        # 入力データの生成
        input = cv2.resize(cropped, (image_size, image_size))
        input_data = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input_data = input_data.astype(np.float32) / 256
        input_data = np.expand_dims(input_data, 0)
        # print(input_data)
        print(input_data.shape)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))
        # 推論の実行
        import time
        st = time.time()
        interpreter.invoke()
        print("el: ", time.time() - st)

        output_data = interpreter.get_tensor(output_details[-1]['index'])
        pre_landmark = output_data.reshape(-1, 2) * [image_size, image_size]

        for (x, y) in pre_landmark.astype(np.int32):
            cv2.circle(input, (x, y), 1, (0, 0, 255), 2)
        cv2.imwrite('samples/labeled.jpg', input)
        # cv2.imshow('1', cropped)

        # pre_landmark = pre_landmark * [size/image_size, size/image_size] - [dx, dy]
        # for (x, y) in pre_landmark.astype(np.int32):
        #     cv2.circle(image, (x1 + x, y1 + y), 2, (0, 0, 255), 2)
    # image = cv2.resize(image, (width, height))
    # cv2.imwrite('samples/labeled.jpg', image)


if __name__ == '__main__':
    img_path = 'samples/DSC00597.JPG'
    img = np.array(cv2.imread(img_path))
    inference(img)