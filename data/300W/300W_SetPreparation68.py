#-*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import shutil

debug = False
#debug = True


def rotate(angle, center, landmark):
    rad = angle * np.pi / 180.0
    alpha = np.cos(rad)
    beta = np.sin(rad)
    M = np.zeros((2, 3), dtype=np.float32)
    M[0, 0] = alpha
    M[0, 1] = beta
    M[0, 2] = (1-alpha)*center[0] - beta*center[1]
    M[1, 0] = -beta
    M[1, 1] = alpha
    M[1, 2] = beta*center[0] + (1-alpha)*center[1]

    landmark_ = np.asarray([(M[0, 0]*x+M[0, 1]*y+M[0, 2],
                             M[1, 0]*x+M[1, 1]*y+M[1, 2]) for (x, y) in landmark])
    return M, landmark_


class ImageDate():
    def __init__(self, line, imgDir, image_size=112):
        self.image_size = image_size
        line = line.strip().split()
        #0-195: landmark 坐标点  196-199: bbox 坐标点;
        #200: 姿态(pose)         0->正常姿态(normal pose)          1->大的姿态(large pose)
        #201: 表情(expression)   0->正常表情(normal expression)    1->夸张的表情(exaggerate expression)
        #202: 照度(illumination) 0->正常照明(normal illumination)  1->极端照明(extreme illumination)
        #203: 化妆(make-up)      0->无化妆(no make-up)             1->化妆(make-up)
        #204: 遮挡(occlusion)    0->无遮挡(no occlusion)           1->遮挡(occlusion)
        #205: 模糊(blur)         0->清晰(clear)                    1->模糊(blur)
        #206: 图片名称
        if len(line) != 147:
            print(line)
        assert(len(line) == 147)
        self.list = line
        self.landmark = np.asarray(list(map(float, line[:136])), dtype=np.float32).reshape(-1, 2)
        self.box = np.asarray(list(map(int, line[136:140])), dtype=np.int32)
        flag = list(map(int, line[140:146]))
        flag = list(map(bool, flag))
        self.pose = flag[0]
        self.expression = flag[1]
        self.illumination = flag[2]
        self.make_up = flag[3]
        self.occlusion = flag[4]
        self.blur = flag[5]
        self.path = os.path.join(imgDir, line[146])
        self.img = None

        self.imgs = []
        self.landmarks = []
        self.boxes = []
        self.euler_angles = []  # 辅助网络训练的三个角度

    # def load_angles(self, landmark):
    #     TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
    #     euler_angles_landmark = []
    #     for index in TRACKED_POINTS:
    #         euler_angles_landmark.append(landmark[2 * index:2 * index + 2])
    #     pitch, yaw, roll = calculate_pitch_yaw_roll(euler_angles_landmark)
    #     return list(pitch, yaw, roll)

    def load_data(self, is_train, repeat, mirror=None):
        if (mirror is not None):
            with open(mirror, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 1
                mirror_idx = lines[0].strip().split(',')
                mirror_idx = list(map(int, mirror_idx))
        xy = np.min(self.landmark, axis=0).astype(np.int32)
        zz = np.max(self.landmark, axis=0).astype(np.int32)
        wh = zz - xy + 1

        center = (xy + wh/2).astype(np.int32)
        img = cv2.imread(self.path)
        boxsize = int(np.max(wh)*1.2)
        xy = center - boxsize//2
        x1, y1 = xy
        x2, y2 = xy + boxsize
        try:
            height, width, _ = img.shape
        except:
            import pdb;pdb.set_trace()
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        # for x, y in (self.landmark + 0.5).astype(np.int32):
        #    cv2.circle(img, (x, y), 1, (0, 0, 255))
        imgT = img[y1:y2, x1:x2]
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
        # if imgT.shape[0] == 0 or imgT.shape[1] == 0:
        #     imgTT = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #     for x, y in (self.landmark+0.5).astype(np.int32):
        #         cv2.circle(imgTT, (x, y), 1, (0, 0, 255))
        #     cv2.imshow('0', imgTT)
        #     if cv2.waitKey(0) == 27:
        #         exit()

        if is_train:
            imgT = cv2.resize(imgT, (self.image_size, self.image_size))
        landmark = (self.landmark - xy)/boxsize

        assert (landmark >= 0).all(), str(landmark) + str([dx, dy])
        assert (landmark <= 1).all(), str(landmark) + str([dx, dy])
        self.imgs.append(imgT)
        self.landmarks.append(landmark)

        if is_train:
            while len(self.imgs) < repeat:
                angle = np.random.randint(-20, 20)
                cx, cy = center
                cx = cx + int(np.random.randint(-boxsize*0.1, boxsize*0.1))
                cy = cy + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
                M, landmark = rotate(angle, (cx, cy), self.landmark)

                imgT = cv2.warpAffine(img, M, (int(img.shape[1]*1.1), int(img.shape[0]*1.1)))
                # for x, y in (landmark + 0.5).astype(np.int32):
                #    cv2.circle(imgT, (x, y), 1, (0, 0, 255))
                wh = np.ptp(landmark, axis=0).astype(np.int32) + 1
                size = np.random.randint(int(np.min(wh)), np.ceil(np.max(wh) * 1.25))
                xy = np.asarray((cx - size // 2, cy - size//2), dtype=np.int32)
                landmark = (landmark - xy) / size
                if (landmark < 0).any() or (landmark > 1).any():
                    continue

                x1, y1 = xy
                x2, y2 = xy + size
                height, width, _ = imgT.shape
                dx = max(0, -x1)
                dy = max(0, -y1)
                x1 = max(0, x1)
                y1 = max(0, y1)

                edx = max(0, x2 - width)
                edy = max(0, y2 - height)
                x2 = min(width, x2)
                y2 = min(height, y2)

                imgT = imgT[y1:y2, x1:x2]
                if (dx > 0 or dy > 0 or edx >0 or edy > 0):
                    imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

                imgT = cv2.resize(imgT, (self.image_size, self.image_size))

                if mirror is not None and np.random.choice((True, False)):
                    landmark[:, 0] = 1 - landmark[:, 0]
                    landmark = landmark[mirror_idx]
                    imgT = cv2.flip(imgT, 1)
                self.imgs.append(imgT)
                self.landmarks.append(landmark)

    def save_data(self, path, prefix):
        attributes = [self.pose, self.expression, self.illumination, self.make_up, self.occlusion, self.blur]
        attributes = np.asarray(attributes, dtype=np.int32)
        attributes_str = ' '.join(list(map(str, attributes)))
        labels = []
        for i, (img, lanmark) in enumerate(zip(self.imgs, self.landmarks)):
            save_path = os.path.join(path, prefix+'_'+str(i)+'.png')
            assert lanmark.shape == (68, 2)
            assert not os.path.exists(save_path), save_path
            cv2.imwrite(save_path, img)
            landmark_str = ' '.join(list(map(str, lanmark.reshape(-1).tolist())))
            label = '{} {} {} \n'.format(save_path, landmark_str, attributes_str)
            labels.append(label)
        return labels


def get_dataset_list(imgDir, outDir, landmarkDir, is_train):
    with open(landmarkDir, 'r') as f:
        lines = f.readlines()
        print(len(lines))
        labels = []
        save_img = os.path.join(outDir, 'imgs')
        if not os.path.exists(save_img):
            os.mkdir(save_img)

        if debug:
            lines = lines[:100]
        for i, line in enumerate(lines):
            Img = ImageDate(line, imgDir)
            img_name = Img.path
            Img.load_data(is_train, 10, Mirror_file)
            _, filename = os.path.split(img_name)
            filename, _ = os.path.splitext(filename)
            label_txt = Img.save_data(save_img, '300W_' + str(i)+'_' + filename)
            labels.append(label_txt)
            if ((i + 1) % 100) == 0:
                print('file: {}/{}'.format(i+1, len(lines)))

    with open(os.path.join(outDir, 'list.txt'), 'w') as f:
        for label in labels:
            f.writelines(label)


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    imageDirs = 'raw'
    Mirror_file = 'annotations/Mirror68.txt'

    landmarkDirs = ['annotations/list_68pt_rect_attr_train_test/list_68pt_rect_attr_train.txt',
                    'annotations/list_68pt_rect_attr_train_test/list_68pt_rect_attr_test.txt']

    outDirs = ['train_data', 'test_data']
    for landmarkDir, outDir in zip(landmarkDirs, outDirs):
        outDir = os.path.join(root_dir, outDir)
        print(outDir)
        if os.path.exists(outDir):
            shutil.rmtree(outDir)
        os.mkdir(outDir)
        if 'list_68pt_rect_attr_test.txt' in landmarkDir:
            is_train = False
        else:
            is_train = True
        imgs = get_dataset_list(imageDirs, outDir, landmarkDir, is_train)
    print('end')