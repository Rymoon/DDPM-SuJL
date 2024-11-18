"""
Tensorflow 115

based on "apps/reference/ddpm2". Seek for good performance.
"""
import os 
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2

from ren_utils.rennet import call_by_inspect 

import sjldpm as pkg
cfn = Path(__file__).stem
cfp = Path(__file__).parent

def imwrite(path, figure):
    """归一化到了[-1, 1]的图片矩阵保存为图片
    """
    print("Imwrite ...\n")
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype('uint8')
    cv2.imwrite(path, figure)

def data_generator(dataset,*, batch_size, T, bar_alpha, bar_beta):
    """
    `dataset`, allow:
    * dataset[i] -> ndarray () 
    * len(dataset)-> int
    """
    assert hasattr(dataset,"__getitem__")
    assert hasattr(dataset,"__len__")
    batch_imgs = []
    while True:
        for i in np.random.permutation(len(dataset)):
            batch_imgs.append(dataset[i])
            if len(batch_imgs) == batch_size:
                batch_imgs = np.array(batch_imgs)
                batch_steps = np.random.choice(T, batch_size)
                batch_bar_alpha = bar_alpha[batch_steps][:, None, None, None]
                batch_bar_beta = bar_beta[batch_steps][:, None, None, None]
                batch_noise = np.random.randn(*batch_imgs.shape)
                batch_noisy_imgs = batch_imgs * batch_bar_alpha + batch_noise * batch_bar_beta
                yield [batch_noisy_imgs, batch_steps[:, None]], batch_noise
                batch_imgs = []
    
def sample(model, path=None, n=4, z_samples=None, t0=0, *, img_size, beta, bar_beta, alpha, sigma, T ):
    """随机采样函数
    """
    print("Start sampling ...\n")
    if z_samples is None:
        z_samples = np.random.randn(n**2, img_size, img_size, 3)
    else:
        z_samples = z_samples.copy()
    for t in tqdm(range(t0, T), ncols=0):
        t = T - t - 1
        bt = np.array([[t]] * z_samples.shape[0])
        z_samples -= beta[t]**2 / bar_beta[t] * model.predict([z_samples, bt])
        z_samples /= alpha[t]
        z_samples += np.random.randn(*z_samples.shape) * sigma[t]
    x_samples = np.clip(z_samples, -1, 1)
    if path is None:
        return x_samples
    figure = np.zeros((img_size * n, img_size * n, 3))
    for i in range(n):
        for j in range(n):
            digit = x_samples[i * n + j]
            figure[i * img_size:(i + 1) * img_size,
                   j * img_size:(j + 1) * img_size] = digit
    imwrite(path, figure)
    
def get_config(config_name:str):
    """基本配置, return as dict
    """
    if config_name == "sjl_64":
        resize_size = (64,64)  # 如果只想快速实验，可以改为64
        img_size = resize_size[0] # Assume Square
        batch_size = 16  # 如果显存不够，可以降低为16，但不建议低于16 # 16 for 24GB
        embedding_size = 128
        channels = [1, 1, 2, 2, 4, 4]
        blocks = 2  # 如果显存不够，可以降低为1

        # 超参数选择
        T = 1000
        alpha = np.sqrt(1 - 0.02 * np.arange(1, T + 1) / T)
        beta = np.sqrt(1 - alpha**2)
        bar_alpha = np.cumprod(alpha)
        bar_beta = np.sqrt(1 - bar_alpha**2)
        sigma = beta.copy()
        # sigma *= np.pad(bar_beta[:-1], [1, 0]) / bar_beta
    elif config_name == "sjl_128":
        resize_size = (128,128)  # 如果只想快速实验，可以改为64
        img_size = resize_size[0] 
        batch_size = 16  # 如果显存不够，可以降低为16，但不建议低于16 # 16 for 24GB
        embedding_size = 128
        channels = [1, 1, 2, 2, 4, 4]
        blocks = 2  # 如果显存不够，可以降低为1

        # 超参数选择
        T = 1000
        alpha = np.sqrt(1 - 0.02 * np.arange(1, T + 1) / T)
        beta = np.sqrt(1 - alpha**2)
        bar_alpha = np.cumprod(alpha)
        bar_beta = np.sqrt(1 - bar_alpha**2)
        sigma = beta.copy()
        # sigma *= np.pad(bar_beta[:-1], [1, 0]) / bar_beta
    elif config_name == "sjl_256":
        resize_size = (256,256)  # 如果只想快速实验，可以改为64
        img_size = resize_size[0] 
        batch_size = 16  # 如果显存不够，可以降低为16，但不建议低于16 # 16 for 24GB
        embedding_size = 128
        channels = [1, 1, 2, 2, 4, 4]
        blocks = 1  # 如果显存不够，可以降低为1

        # 超参数选择
        T = 1000
        alpha = np.sqrt(1 - 0.02 * np.arange(1, T + 1) / T)
        beta = np.sqrt(1 - alpha**2)
        bar_alpha = np.cumprod(alpha)
        bar_beta = np.sqrt(1 - bar_alpha**2)
        sigma = beta.copy()
        # sigma *= np.pad(bar_beta[:-1], [1, 0]) / bar_beta
    elif config_name == "mnist":
        resize_size = (32,32)  
        img_size = resize_size[0] 
        batch_size = 32  # 如果显存不够，可以降低为16，但不建议低于16 # 16 for 24GB
        embedding_size = 128
        channels = [1, 1, 2, 4]
        blocks = 2  # 如果显存不够，可以降低为1

        # 超参数选择
        T = 1000
        s0,s1 = 0.01,20
        t = (np.arange(1, T + 1)-1) / (T-1)
        sigma = s0 * ((s1/s0) ** (t))

        # sigma *= np.pad(bar_beta[:-1], [1, 0]) / bar_beta
    else:
        raise Exception(config_name)
    _config = dict(locals())
    config = {k:v for k,v in _config.items() if k[0]!="_"}
    
        
    return config

from sjldpm.apps.reference.ddpm2_h import imread,list_pictures
from ren_utils import rennet
class CelebAHQ:
    def __init__(self,image_resize):
        img_path_list = []
        for _name in ["CelebAHQ256","CelebAHQ256_valid"]:
            img_path_list += list_pictures(rennet.datasets[_name]["imgs"],rennet.datasets[_name]["suffix"] )
        np.random.shuffle(img_path_list)
        self.img_path_list = img_path_list
        
        self.image_resize = image_resize
    def __getitem__(self,i):
        return imread( self.img_path_list[i], self.image_resize)
    
    def __len__(self):
        return len(self.img_path_list)

class STL10:
    def __init__(self,image_resize,crop_size=None):
        c =3
        h=w=96
        proot = Path(Path(pkg.__file__).parent.parent,f"Datasets/STL10/stl10_binary/")

        _x = np.fromfile(Path(proot,"train_X.bin").as_posix(),dtype=np.uint8)
        _x = _x.reshape(-1,c,h,w).transpose(0,3,2,1) #  ... and swap h and w.

        # _y = np.fromfile(Path(proot,"train_y.bin"),dtype=np.uint8)
        
        self.img_ndarray = _x
        
        self.image_resize = image_resize
        self.crop_size = crop_size
    def __getitem__(self,i):
        x = self.img_ndarray[i] # HWC
        height, width = x.shape[:2]
        if self.crop_size is None:
            crop_size = min([height, width])
        else:
            crop_size = min([crop_size, height, width])
        height_x = (height - crop_size + 1) // 2
        width_x = (width - crop_size + 1) // 2
        x = x[height_x:height_x + crop_size, width_x:width_x + crop_size]
        if x.shape[:2] != self.image_resize :
            x = cv2.resize(x, self.image_resize )
        x = x.astype('float32')
        x = x / 255 * 2 - 1
        return x
    
    def __len__(self):
        return len(self.img_ndarray)

import sjldpm
class MNIST:
    root_pkg = Path(sjldpm.__file__).parent
    npz_path = Path(root_pkg,"../Datasets/MNIST/mnist.npz").as_posix()
    def __init__(self,image_resize,digits=[0,1,2,3,4,5,6,7,8,9]):
    
        data_npz  = np.load(self.npz_path)
        x_train = data_npz["x_train"]
        y_train = data_npz["y_train"]
        
        index = []
        for i in range(len(y_train)):
            if y_train[i] in digits:
                index.append(i)
        
        np.random.shuffle(index)
        
        x= (x_train/255)*2-1
        x_= np.zeros((*image_resize,3,len(index)))
        y_ = np.zeros(len(index))
        for i in range(len(index)):
            x_[:,:,0,i]= cv2.resize(x[index[i]],image_resize)
            x_[:,:,1,i]= cv2.resize(x[index[i]],image_resize)
            x_[:,:,2,i]= cv2.resize(x[index[i]],image_resize)
            y_[i] = index[i]
        self.x_ = x_
        self.y_ = y_
        self.index = index
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self,i):
        v = self.x_[:,:,:,i]
        return v

from sjldpm.apps.reference.ddpm2_m import get_model, Trainer
if __name__ == "__main__":
    DEBUG = False
    gpuid = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpuid}"
    
    dm_name = "mnist0246"
    config_name = "mnist"
    
    train_name = f"{pkg.__name__}-{cfn}__{dm_name}__{config_name}" if not DEBUG else f"{pkg.__name__}-DEBUG-{cfn}__{dm_name}__{config_name}" 
    
    log_dir = Path(Path(pkg.__file__).parent.parent,"Results",train_name)
    log_dir .mkdir(parents=True,exist_ok=True)
    _versions = os.listdir(Path(log_dir))
    if len(_versions) ==0:
        log_dir = Path(log_dir,"version_0")
        log_dir.mkdir(parents=True,exist_ok=True)
    else:
        _versions.sort(key = lambda s:int(s.split("_")[1])) # ascend
        next_version_int = int(_versions[-1].split("_")[1])+1
        log_dir = Path(log_dir,f"version_{next_version_int}")
        log_dir.mkdir(parents=True,exist_ok=True)

    config = get_config(config_name)
    if dm_name == "celebahq":
        dataset = CelebAHQ(config["resize_size"])
    elif dm_name =="stl10":
        dataset = STL10(config["resize_size"])
    elif dm_name =="mnist0246":
        dataset = MNIST(config["resize_size"],[0,2,4,6])
    else:
        raise Exception(dm_name)
    dataloader = call_by_inspect(data_generator, config,  dataset = dataset)
    
    model = call_by_inspect( get_model, config)
    print(" - log dir:",log_dir.as_posix())
    trainer = Trainer(model, model.optimizer, 
                    lambda model, path: sample(model, path, n=4, z_samples=None, t0=0, img_size=config["img_size"], beta=config["beta"], bar_beta=config["bar_beta"], alpha=config["alpha"], sigma=config["sigma"], T=config["T"]), 
                    log_dir)
    model.fit(
        dataloader,
        steps_per_epoch=2000 if not DEBUG else 8, # ori:2000
        epochs=1000 if not DEBUG else 14,  # 只是预先设置足够多的epoch数，可以自行Ctrl+C中断
        callbacks=[trainer]
    )