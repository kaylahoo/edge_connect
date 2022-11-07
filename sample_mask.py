import os
import random
import shutil
import argparse


def getRandomList(start, get_len, get_size):
    a = []
    b = []
    c = []

    for i in range(start, start + get_len):
        a.append(i)

    for i in range(start, start + get_len):
        index = random.randint(0, len(a) - 1)
        b.append(a[index])
        a.pop(index)

    for j in range(get_size):
        c.append(b[j])

    return c


def get_mask(name, ratio, get_len, get_num):
    # mask_path = "../preImage"  # mask绝对路径
    mask_path = "/home/lab265/lab265/csy/datasets/irregular_mask/masks"  # mask绝对路径
    dst_path = "/home/lab265/lab265/csy/datasets/irregular_mask"
    f = sorted(os.listdir(mask_path))
    file_list = []
    get_file_len = get_len  # 每间隔n个
    get_file_num = get_num  # 取n个样本
    iter_num = 0
    for i, n in enumerate(f):

        if name == 'celeba':
            dir_name = 'celeba' + str(iter_num) + '-' + str(iter_num + ratio)
        else:
            dir_name = 'psv' + str(iter_num) + '-' + str(iter_num + ratio)

        if (i + 1) % get_file_len == 0:
            index = getRandomList((i + 1) - get_file_len, get_file_len, get_file_num)
            os.makedirs(dst_path + '/' + dir_name)
            for j in index:
                file_list.append(mask_path + f[j])
                shutil.copy(mask_path + '/' + f[j], dst_path + '/' + dir_name + "/" + f[j])

            iter_num = iter_num + ratio


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='celeba')
parser.add_argument('--ratio', type=int, default=10)
parser.add_argument('--get_len', type=int, default=2000)
parser.add_argument('--get_num', type=int, default=100)
args = parser.parse_args()

get_mask(args.name, args.ratio, args.get_len, args.get_num)
