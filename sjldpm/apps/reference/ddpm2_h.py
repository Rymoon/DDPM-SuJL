
import cv2
import numpy as np
import warnings
from pathlib import Path
from keras_preprocessing.image import list_pictures

warnings.filterwarnings("ignore")  # 忽略keras带来的满屏警告


def create_next_version_dir(path):
    # Find all existing version directories
    version_dirs = list(Path(path).glob('version_*'))

    # If there are no existing version directories, create version_0
    if not version_dirs:
        next_version_dir = Path(path, 'version_0')
        next_version_dir.mkdir(exist_ok=True)
        return str(next_version_dir)

    # Otherwise, find the highest existing version index and increment it
    highest_index = max([int(d.name.split('_')[1]) for d in version_dirs])
    next_version_index = highest_index + 1
    next_version_dir = Path(path, f'version_{next_version_index}')
    next_version_dir.mkdir(exist_ok=True)
    return str(next_version_dir)

def imread(f, img_size, crop_size=None):
    """读取图片
    """
    x = cv2.imread(f)
    height, width = x.shape[:2]
    if crop_size is None:
        crop_size = min([height, width])
    else:
        crop_size = min([crop_size, height, width])
    height_x = (height - crop_size + 1) // 2
    width_x = (width - crop_size + 1) // 2
    x = x[height_x:height_x + crop_size, width_x:width_x + crop_size]
    if x.shape[:2] != (img_size, img_size):
        x = cv2.resize(x, (img_size, img_size))
    x = x.astype('float32')
    x = x / 255 * 2 - 1
    return x


def imwrite(path, figure):
    """归一化到了[-1, 1]的图片矩阵保存为图片
    """
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype('uint8')
    cv2.imwrite(path, figure)



