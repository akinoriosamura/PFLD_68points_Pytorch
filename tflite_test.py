from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('pid: {}     GPU: {}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))
import numpy as np
import cv2
import time
from RetinaFaceMaster.test import predict
from mtcnn.detect_face import MTCNN
from model2 import MobileNetV2, BlazeLandMark, MyResNest50
import torch
import shutil
from tensorflow.lite.python.interpreter import Interpreter


def main():
    model_file = 'models2/wflw_moruhard_grow_68_Res50/all_model_85_opt.tflite'
    images_dir = './testdata/'  #'./data/test_data/imgs/'
    image_size = 128  # 112
    labeled_dir = './show_labeled_wflw_moruhard_grow_68_Res50'
    if not os.path.exists(labeled_dir):
        os.mkdir(labeled_dir)
    else:
        shutil.rmtree(labeled_dir)
        os.mkdir(labeled_dir)

    # インタプリタの生成
    interpreter = Interpreter(model_path=model_file)
    interpreter.allocate_tensors()
    # 入力情報と出力情報の取得
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image_files = os.listdir(images_dir)
    total_time = 0
    total_num = 0
    for index, image_file in enumerate(image_files):
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        if image is None:
            print(image_path)
            continue
        # boxes = mtcnn.predict(image)
        # image = cv2.resize(image, (width//2, height//2))
        boxes, _ = predict(image)
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
            cropped = cv2.resize(cropped, (image_size, image_size))
            cv2.imwrite(os.path.join("samples", image_file[:-4] + ".png"), cropped)
            input = cv2.resize(cropped, (image_size, image_size))
            input = cv2.cvtColor(input, cv2.COLOR_RGB2BGR)
            input = input.astype(np.float32) / 256.0
            input = np.expand_dims(input, 0)
            # input = torch.Tensor(input.transpose((0, 3, 1, 2)))
            interpreter.set_tensor(input_details[0]['index'], input)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))
            import time
            st = time.time()
            interpreter.invoke()
            total_time += time.time() - st
            total_num += 1
            output_data = interpreter.get_tensor(output_details[-1]['index'])
            pre_landmark = output_data.reshape(-1, 2) * [image_size, image_size]

            # for (x, y) in pre_landmark.astype(np.int32):
            #     cv2.circle(cropped, (x, y), 1, (0, 0, 255), 2)
            # cv2.imshow('1', cropped)

            pre_landmark = pre_landmark * [size/image_size, size/image_size] - [dx, dy]
            for (x, y) in pre_landmark.astype(np.int32):
                cv2.circle(image, (x1 + x, y1 + y), 2, (0, 0, 255), 2)
        image = cv2.resize(image, (width, height))
        test_f_name = os.path.basename(image_file)
        cv2.imwrite(os.path.join(labeled_dir, test_f_name + "_labeled.jpg"), image)
        print("save labeled: ", os.path.join(labeled_dir, test_f_name + "_labeled.jpg"))
        # image = cv2.resize(image, (width, height))
        # cv2.imshow('0', image)
        # if cv2.waitKey(0) == 27:
        #     break
    ave = total_time / total_num
    print("ave: ", ave)

if __name__ == '__main__':
    main()
