import os
import argparse
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to the dataset')
parser.add_argument('--output', type=str, help='path to the file list')
args = parser.parse_args()

ext = {'.JPG', '.JPEG', '.PNG', '.TIF', 'TIFF'}

images = []
img1 = []
img2 = []
for root, dirs, files in os.walk(args.path):
    print('loading ' + root)
    for file in files:
        if os.path.splitext(file)[1].upper() in ext:
            path = os.path.join(root, file)
            img = Image.open(path)
            height, weight = img.size
            img_arr = np.array(img)
            ratio = int(np.count_nonzero(img_arr) / (height * weight) * 10)
            if ratio >= 0.2 and ratio < 0.4:
                img1.append(path)
            if ratio >= 0.4:
                img2.append(path)
            images.append(path)

images = sorted(images)
img1 = sorted(img1)
img2 = sorted(img2)
np.savetxt(args.output + '.flist', images, fmt='%s')
np.savetxt(args.output + '_1.flist', img1, fmt='%s')
np.savetxt(args.output + '_2.flist', img2, fmt='%s')
