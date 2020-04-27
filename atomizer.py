import cv2
import argparse
from timeit import default_timer as timer
import os
import numpy as np

'''
parse the arguments from terminal
'''
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='./', help='path to input iamges')
parser.add_argument('--output', type=str, default='./result/', help='path to save result')
parser.add_argument('--transpancy', type=float, default=0.75, help='transpancy of fog or mog')
parser.add_argument('--alpha', type=float, default=0.06, help='control coefficient alpha')
parser.add_argument('--beta', type=float, default=0.2, help='control corfficient beta')
parser.add_argument('--gamma', type=float, default=-15, help='control coefficient gamma')
parser.add_argument('--out_size', type=tuple, default=(256, 256), help='the size of output')
parser.add_argument('--center', type=tuple, default=(0.25, 0.5),
                    help='(h,w) is the ratio for locating,the center position will be (h*height,w*width)')
parser.add_argument('--keep_name', type=bool, default=False, help='whether to keep original filename after output')
option = parser.parse_args()
# receive parameters
input = option.input
output = option.output
transpancy = option.transpancy
alpha = option.alpha
beta = option.beta
gamma = option.gamma
out_size = option.out_size
center = option.center
keep_name = option.keep_name

'''
auxiliary functions
'''


def save_result(output, id, img):
    if id < 10:
        new_name = '0000%d.png' % id
    elif id < 100:
        new_name = '000%d.png' % id
    elif id < 1000:
        new_name = '00%d.png' % id
    elif id < 10000:
        new_name = '0%d.png' % id
    else:
        new_name = '%id.png' % id
    cv2.imwrite('%s/%s' % (output, new_name), img)


def info(t_start, t_stop, num_img):
    print('average time of processing single image: %.3f ms' % ((t_stop - t_start) * 1000 / num_img))
    if num_img == 0:
        print('Nothing is done in this round, check the path of input')
    else:
        print('Operation is done.')
        print('Total %d images have beed processed.' % num_img)


def author():
    print('Author@Matthew LC Zheng\n'
          'Organization@UESTC\n'
          'Project@Bachelor dissertation: Domain adaptation for isntance segmentation\n'
          'Repository@https://github.com/matthew-lc-zheng/adaptive-seg\n'
          'License@Apache-2.0')
    
    
'''
main part of atomizer
'''


def atomizer():
    C=[]
    C.append(center[0] * out_size[0])
    C.append(center[1] * out_size[1])
    id = 0
    for _, _, filenames in os.walk(input):
        for filename in filenames:
            img = cv2.imread('%s/%s' % (input, filename))
            img = cv2.resize(img, out_size, interpolation=cv2.INTER_CUBIC)
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    xi = np.exp(beta * (alpha * np.sqrt((i - C[0]) ** 2 + (j - C[1]) ** 2) + gamma))
                    img[i, j, :] = img[i, j, :] * xi + transpancy * (1 - xi)*255
            id += 1
            if keep_name:
                cv2.imwrite('%s/%s' % (output, filename), img)
            else:
                save_result(output, id, img)
    return id


if __name__ == '__main__':
    author()
    t_start = timer()
    num_img = atomizer()
    t_stop = timer()
    info(t_start, t_stop, num_img)
