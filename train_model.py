# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('pid: {}     GPU: {}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))

import torch
import torchvision as tv
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import cv2
import argparse
import sys
import re
import time
from generate_data import DataSet
from model2 import MobileNetV2, BlazeLandMark, AuxiliaryNet, WingLoss, EfficientLM, HighResolutionNet, MyResNest50, MyResNest200, MyResNest269
from utils import train_model
from euler_angles_utils import calculate_pitch_yaw_roll


def get_euler_angle_weights(landmarks_batch, euler_angles_pre, device, num_label):
    if num_label == 68:
        TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
    elif num_label == 98:
        TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
    else:
        exit()

    euler_angles_landmarks = []
    landmarks_batch = landmarks_batch.numpy()
    for index in TRACKED_POINTS:
        euler_angles_landmarks.append(landmarks_batch[:, 2 * index:2 * index + 2])
    euler_angles_landmarks = np.asarray(euler_angles_landmarks).transpose((1, 0, 2)).reshape((-1, 28))

    euler_angles_gt = []
    for j in range(euler_angles_landmarks.shape[0]):
        pitch, yaw, roll = calculate_pitch_yaw_roll(euler_angles_landmarks[j])
        euler_angles_gt.append((pitch, yaw, roll))
    euler_angles_gt = np.asarray(euler_angles_gt).reshape((-1, 3))

    euler_angles_gt = torch.Tensor(euler_angles_gt).to(device)
    euler_angle_weights = 1 - torch.cos(torch.abs(euler_angles_gt - euler_angles_pre))
    euler_angle_weights = torch.sum(euler_angle_weights, 1)

    return euler_angle_weights


def main(args):
    print(args)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data_transforms = tv.transforms.Compose([
        tv.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        tv.transforms.Resize((args.image_size, args.image_size)),
        tv.transforms.ToTensor()
    ])
    test_data_transforms = tv.transforms.Compose([
        tv.transforms.Resize((args.image_size, args.image_size)),
        tv.transforms.ToTensor()
    ])
    train_dataset = DataSet(args.num_label, args.file_list, args.image_channels, args.image_size, transforms=train_data_transforms,
                            is_train=True)
    test_dataset = DataSet(args.num_label, args.test_list, args.image_channels, args.image_size, transforms=test_data_transforms,
                           is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)

    model_dir = args.model_dir
    print(model_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    print('Total number of examples: {}'.format(len(train_dataset)))
    print('Test number of examples: {}'.format(len(test_dataset)))
    print('Model dir: {}'.format(model_dir))

    if args.save_image_example:
        save_image_example(train_loader, args)
    
    #MobileNetV2
    # coefficient = 0.25
    # num_of_channels = [int(64 * coefficient), int(128 * coefficient), int(16 * coefficient), int(32 * coefficient), int(128 * coefficient)]
    # model = MobileNetV2(num_of_channels=num_of_channels, nums_class=args.num_label)  # model
    # auxiliary_net = AuxiliaryNet(input_channels=num_of_channels[0])
    
    #BlazeLandMark
    #model = BlazeLandMark(nums_class=args.num_label)
    #auxiliary_net = AuxiliaryNet(input_channels=48, first_conv_stride=2)
    
    """
        compound_coef=0 : efficientNet-b0;
        compound_coef=1 : efficientNet-b1;
        compound_coef=2 : efficientNet-b2;
        compound_coef=3 : efficientNet-b3;
        compound_coef=4 : efficientNet-b4;
        compound_coef=5 : efficientNet-b5;
        compound_coef=6 : efficientNet-b6;
        compound_coef=7 : efficientNet-b7;
    """
    #model = EfficientLM(nums_class=args.num_label, compound_coef=0)
    #auxiliary_net = AuxiliaryNet(input_channels=model.p8_outchannels, first_conv_stride=2)
    
    #model = HighResolutionNet(nums_class=args.num_label)
    #auxiliary_net = AuxiliaryNet(input_channels=64, first_conv_stride=2)
    
    # model = MyResNest50(nums_class=args.num_label * 2)
    # auxiliary_net = AuxiliaryNet(input_channels=64, first_conv_stride=2)
    model = MyResNest200(nums_class=args.num_label * 2)
    auxiliary_net = AuxiliaryNet(input_channels=128, first_conv_stride=2)
    # model = MyResNest269(nums_class=args.num_label * 2)
    # auxiliary_net = AuxiliaryNet(input_channels=128, first_conv_stride=2)

    start_epoch = 0

    if args.pretrained_model:
        pretrained_model = args.pretrained_model
        if args.all_model:
            print('load all model, model graph and weight included!')
            if not os.path.isdir(pretrained_model):
                print('Restoring pretrained model: {}'.format(pretrained_model))
                model = torch.load(pretrained_model)
                # import pdb;pdb.set_trace()
                start_epoch = int(re.findall(r'\d+', os.path.basename(pretrained_model))[-1])
            else:
                print('Model directory: {}'.format(pretrained_model))
                files = os.listdir(pretrained_model)
                assert len(files) == 1 and files[0].split('.')[-1] in ['pt', 'pth']
                model_path = os.path.join(pretrained_model, files[0])
                print('Model name:{}'.format(files[0]))
                model = torch.load(model_path)
                start_epoch = int(re.findall(r'\d+', os.path.basename(model_path))[-1])
        else:
            if not os.path.isdir(pretrained_model):
                print('Restoring pretrained model: {}'.format(pretrained_model))
                model.load_state_dict(torch.load(pretrained_model))
                start_epoch = int(re.findall(r'\d+', os.path.basename(pretrained_model))[-1])
            else:
                print('Model directory: {}'.format(pretrained_model))
                files = os.listdir(pretrained_model)
                assert len(files) == 1 and files[0].split('.')[-1] in ['pt', 'pth']
                model_path = os.path.join(pretrained_model, files[0])
                print('Model name:{}'.format(files[0]))
                model.load_state_dict(torch.load(model_path))
                start_epoch = int(re.findall(r'\d+', os.path.basename(model_path))[-1])
        test(test_loader, model, args, device)

    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    model.to(device)
    auxiliary_net.to(device)
    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': auxiliary_net.parameters()}], lr=args.learning_rate, weight_decay=args.weight_decay)  # optimizer
    lr_epoch = args.lr_epoch.strip().split(',')
    lr_epoch = list(map(int, lr_epoch))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_epoch, gamma=0.1)
    wing_loss = WingLoss(10.0, 2.0)

    print('Running train.')
    print('start_epoch: ', start_epoch)

    start_time = time.time()
    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        auxiliary_net.train()

        for i_batch, (images_batch, landmarks_batch, attributes_batch) in enumerate(train_loader):
            images_batch = Variable(images_batch.to(device))
            landmarks_batch = Variable(landmarks_batch)
            pre_landmarks, auxiliary_features = model(images_batch)
            euler_angles_pre = auxiliary_net(auxiliary_features)
            euler_angle_weights = get_euler_angle_weights(landmarks_batch, euler_angles_pre, device, args.num_label)
            loss = wing_loss(landmarks_batch.to(device), pre_landmarks, euler_angle_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ((i_batch + 1) % 100) == 0 or (i_batch + 1) == len(train_loader):
                Epoch = 'Epoch:[{:<4}][{:<4}/{:<4}]'.format(epoch, i_batch + 1, len(train_loader))
                Loss = 'Loss: {:2.3f}'.format(loss.item())
                trained_sum_iters = len(train_loader) * epoch + i_batch + 1
                average_time = (time.time() - start_time) / trained_sum_iters
                remain_time = average_time * (len(train_loader) * args.max_epoch - trained_sum_iters) / 3600
                print('{}\t{}\t lr {:2.3}\t average_time:{:.3f}s\t remain_time:{:.3f}h'.format(Epoch, Loss,
                                                                                               optimizer.param_groups[0]['lr'],
                                                                                               average_time,
                                                                                               remain_time))
        scheduler.step()
        # save model
        """
        checkpoint_path = os.path.join(model_dir, 'model_'+str(epoch)+'.pth')
        if args.all_model:
            torch.save(model, checkpoint_path)
        else:
            torch.save(model.state_dict(), checkpoint_path)
        """
        # save all model
        checkpoint_path = os.path.join(model_dir, 'all_model_'+str(epoch)+'.pth')
        torch.save(model, checkpoint_path)
        # save state dict
        checkpoint_path = os.path.join(model_dir, 'model_'+str(epoch)+'.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print("finish save pth model file")

        if epoch % 3 == 0:
            test(test_loader, model, args, device)


def test(test_loader, model, args, device):

    model.eval()
    sample_path = os.path.join(args.model_dir, 'HeatMaps')
    if not os.path.exists(sample_path):
        os.mkdir(sample_path)

    test_num = 0
    test_sum_time = 0
    ave_test_time = 0
    loss_sum = 0
    landmark_error = 0
    landmark_01_num = 0
    for i_batch, (images_batch, landmarks_batch, attributes_batch) in enumerate(test_loader):
        images_batch = Variable(images_batch.to(device))
        landmarks_batch = Variable(landmarks_batch)
        # inference
        test_num += 1
        st = time.time()
        pre_landmarks, euler_angles_pre = model(images_batch)
        test_sum_time += time.time() - st

        images_batch = images_batch.cpu().numpy()
        landmarks_batch = landmarks_batch.numpy()
        pre_landmarks = pre_landmarks.cpu().detach().numpy()

        diff = pre_landmarks - landmarks_batch
        loss = np.sum(diff * diff)
        loss_sum += loss

        for k in range(pre_landmarks.shape[0]):
            error_all_points = 0
            for count_point in range(pre_landmarks.shape[1] // 2):  # num points
                error_diff = pre_landmarks[k][(count_point * 2):(count_point * 2 + 2)] - landmarks_batch[k][
                                                                                         (count_point * 2):(
                                                                                         count_point * 2 + 2)]
                error = np.sqrt(np.sum(error_diff * error_diff))
                error_all_points += error
            interocular_distance = np.sqrt(np.sum(pow((landmarks_batch[k][72:74] - landmarks_batch[k][90:92]), 2)))
            error_norm = error_all_points / (interocular_distance * args.num_label)
            landmark_error += error_norm
            if error_norm >= 0.1:
                landmark_01_num += 1

        if i_batch == 0:
            image_save_path = os.path.join(sample_path, 'img')
            if not os.path.exists(image_save_path):
                os.mkdir(image_save_path)
            images_batch = images_batch.transpose((0, 2, 3, 1))
            for j in range(images_batch.shape[0]):  # batch_size
                image = images_batch[j] * 256
                image = image[:, :, ::-1]

                image_i = image.copy()
                pre_landmark = pre_landmarks[j]
                h, w, _ = image_i.shape
                pre_landmark = pre_landmark.reshape(-1, 2) * [w, h]
                for (x, y) in pre_landmark.astype(np.int32):
                    cv2.circle(image_i, (x, y), 1, (0, 0, 255))
                landmark = landmarks_batch[j].reshape(-1, 2) * [w, h]
                for (x, y) in landmark.astype(np.int32):
                    cv2.circle(image_i, (x, y), 1, (255, 0, 0))
                image_save_name = os.path.join(image_save_path, '{}.jpg'.format(j))
                cv2.imwrite(image_save_name, image_i)

    ave_test_time = test_sum_time / test_num
    print("test_num: ", test_num)
    print("ave test time: ", ave_test_time)

    loss = loss_sum / (len(test_loader) * args.batch_size)
    print('Test epochs: {}\tLoss {:2.3f}'.format(len(test_loader), loss))

    print('mean error and failure rate')
    landmark_error_norm = landmark_error / (len(test_loader) * args.batch_size)
    error_str = 'mean error : {:2.3f}'.format(landmark_error_norm)
    failure_rate = landmark_01_num / (len(test_loader) * args.batch_size)
    failure_rate_str = 'failure rate: L1 {:2.3f}'.format(failure_rate)
    print(error_str + '\n' + failure_rate_str + '\n')


def save_image_example(train_loader, args):
    save_nbatch = 10
    save_path = os.path.join(args.model_dir, 'image_example')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i_batch, (images, landmarks, attributes) in enumerate(train_loader):
        images = images.numpy()
        landmarks = landmarks.numpy()
        images = images.transpose((0, 2, 3, 1))
        for i in range(images.shape[0]):
            img = images[i] * 255
            img = img.astype(np.uint8)
            if args.image_channels == 1:
                img = np.concatenate((img, img, img), axis=2)
            else:
                img = img[:, :, ::-1].copy()
            land = landmarks[i].reshape(-1, 2) * img.shape[:2]
            for x, y in land.astype(np.int32):
                cv2.circle(img, (x, y), 1, (0, 0, 255))
            save_name = os.path.join(save_path, '{}_{}.jpg'.format(i_batch, i))
            cv2.imwrite(save_name, img)
        if i_batch == save_nbatch:
            break


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_list', type=str, default='data/train_data/list.txt')
    parser.add_argument('--test_list', type=str, default='data/test_data/list.txt')
    parser.add_argument('--loss_log_dir', type=str, default='./train_loss_log/')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--image_channels', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--pretrained_model', type=str, default='')
    parser.add_argument('--model_dir', type=str, default='models2/model_test')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--lr_epoch', type=str, default='10,20,50,100,200,500')
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--level', type=str, default='L5')
    parser.add_argument('--save_image_example', action='store_false', default=True)
    parser.add_argument('--all_model', action='store_true', default=True)
    parser.add_argument('--num_label', type=int, default=68)

    return parser.parse_args(argv)


if __name__ == '__main__':
    print(sys.argv)
    main(parse_arguments(sys.argv[1:]))

