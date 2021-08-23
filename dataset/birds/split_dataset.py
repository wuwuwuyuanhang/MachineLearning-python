# @Auther : wuwuwu 
# @Time : 2021/8/21
# @File : split_dataset.py
# @Description :


import os
from shutil import rmtree, copy
import random
import argparse


def mk_file(file_path):
    if os.path.exists(file_path):
        rmtree(file_path)  # 删除空文件夹
    os.makedirs(file_path)


def train_test_dataset_split(data_path, rate=0.1):
    """
    训练与测试数据集分割
    :param data_path: 数据集文件夹
    :param rate: 测试集占比
    :return:
    """
    root_path = os.path.join(data_path, '..')
    data_class = [cla for cla in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, cla))]

    # 创建对应文件夹，即在 train 和 test 文件夹下，每个类别各一个单独文件夹
    train_path = os.path.join(root_path, 'train')
    test_path = os.path.join(root_path, 'test')
    mk_file(train_path)
    mk_file(test_path)
    for cla in data_class:
        mk_file(os.path.join(train_path, cla))
        mk_file(os.path.join(test_path, cla))

    for cla in data_class:
        cla_path = os.path.join(data_path, cla)
        images = os.listdir(cla_path)
        num = len(images)

        test_image = random.sample(images, int(num * rate))  # 随机分割，返回一个 image_path 列表
        for index, image in enumerate(images):
            if image in test_image:
                # 将分配至验证集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(test_path, cla)
                copy(image_path, new_path)
            else:
                # 将分配至训练集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_path, cla)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
        print()

    print("processing done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help="数据集路径")
    parser.add_argument('--split_rate', type=float, default=0.1, help="测试集比例")
    args = parser.parse_args()
    train_test_dataset_split(data_path=args.data_path, rate=args.split_rate)