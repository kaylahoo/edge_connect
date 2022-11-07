import os
import random
import shutil


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


def get_mask():
    #mask_path = "../preImage"  # mask绝对路径
    mask_path = "/home/lab265/lab265/csy/datasets/irregular_mask/masks"  # mask绝对路径
    dst_path = "/home/lab265/lab265/csy/datasets/irregular_mask"
    f = os.listdir(mask_path)
    file_list = []
    get_file_len = 2000  # 每间隔n个
    get_file_num = 100  # 取n个样本
    iter_num = 0
    for i, n in enumerate(f):
        dir_name = str(iter_num) + '-' + str(iter_num + 10)
        if (i + 1) % get_file_len == 0:
            index = getRandomList((i + 1) - get_file_len, get_file_len, get_file_num)
            os.makedirs(dst_path + '/' + dir_name)
            for j in index:
                file_list.append(mask_path + f[j])
                shutil.copy(mask_path + '/' + f[j], dst_path + '/' + dir_name + "/" + f[j])
            iter_num = iter_num + 10

    print(file_list)


get_mask()
