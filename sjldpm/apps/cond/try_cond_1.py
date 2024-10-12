"""
Tensorflow 1.15

Self contained script.

1. Load ckpt and predict
2. Load one image and use as conds.
"""

import cv2
import numpy as np
import os 
from pathlib import Path
cfn = Path(__file__).stem
cfp = Path(__file__).parent
from tqdm import tqdm, trange
from uitls import call_by_inspect
import sjldpm
from sjldpm.apps.reference.ddpm2_m import get_model

def demo():
    # Create model
    pass

def get_config():
    resize_size = (128,128) 
    img_size = resize_size[0]
    embedding_size = 128
    channels = [1, 1, 2, 2, 4, 4]
    blocks = 2  

    T = 1000
    alpha = np.sqrt(1 - 0.02 * np.arange(1, T + 1) / T)
    beta = np.sqrt(1 - alpha**2)
    bar_alpha = np.cumprod(alpha)
    bar_beta = np.sqrt(1 - bar_alpha**2)
    sigma = beta.copy()

    _config = dict(locals())
    config = {k:v for k,v in _config.items() if k[0]!="_"}
    return config

def sample(model, path=None, n=4, z_samples=None, t0=0, *, img_size, beta, bar_beta, alpha, sigma, T ):
    """
    image in [-1,1]
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
    
def sample_ddim(model,path=None, n=4, z_samples=None, stride=1, eta=1,*,img_size,bar_alpha):
    """随机采样函数
    eta: reletive scale of variance
    stride: jump in time
    """
    return core_sample_ddim(model, path,n,z_samples,stride,eta,img_size = img_size,bar_alpha=bar_alpha,cond_fn=lambda z_samples,bt,bar_beta: z_samples)
    
def core_sample_ddim(model, path=None, n=4, z_samples=None, stride=1, eta=1,*,img_size,bar_alpha,cond_fn):
    """随机采样函数
    eta: reletive scale of variance
    stride: jump in time
    """
    # 采样参数
    bar_alpha_ = bar_alpha[::stride]
    bar_alpha_pre_ = np.pad(bar_alpha_[:-1], [1, 0], constant_values=1)
    bar_beta_ = np.sqrt(1 - bar_alpha_**2)
    bar_beta_pre_ = np.sqrt(1 - bar_alpha_pre_**2)
    alpha_ = bar_alpha_ / bar_alpha_pre_
    sigma_ = bar_beta_pre_ / bar_beta_ * np.sqrt(1 - alpha_**2) * eta
    epsilon_ = bar_beta_ - alpha_ * np.sqrt(bar_beta_pre_**2 - sigma_**2)
    T_ = len(bar_alpha_)
    # 采样过程
    if z_samples is None:
        z_samples = np.random.randn(n**2, img_size, img_size, 3) # Noise
    else:
        z_samples = z_samples.copy()
    for t in tqdm(range(T_), ncols=0):
        t = T_ - t - 1
        bt = np.array([[t * stride]] * z_samples.shape[0])
        z_samples = cond_fn(z_samples,bt,bar_beta_)
        z_samples -= epsilon_[t] * model.predict([z_samples, bt])
        z_samples /= alpha_[t]
        z_samples += np.random.randn(*z_samples.shape) * sigma_[t]
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
    return figure

def imwrite(path, figure):
    """[-1,1] figure -> [0,255]  image to save
    """
    print("Imwrite ...\n")
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype('uint8')
    cv2.imwrite(path, figure)


    
def create_log_dir(pkg,cfn,name):
    log_dir = _get_log_dir(pkg,cfn,name)
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
    return log_dir

def use_log_dir(pkg,cfn,name,version:int):
    log_dir = _get_log_dir(pkg,cfn,name)
    log_dir.mkdir(parents=True,exist_ok=True)
    log_dir = Path(log_dir,f"version_{version}")
    if not log_dir.exists():
        log_dir.mkdir(parents=True,exist_ok=True)
    return log_dir
    
def _get_log_dir(pkg,cfn,name):
    train_name = f"{pkg.__name__}-{cfn}__{name}" 
    log_dir = Path(Path(pkg.__file__).parent.parent,"Results",train_name)
    return log_dir

if __name__ =="__main__":
    gpuid = 3
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpuid}"
    log_dir = use_log_dir(sjldpm,cfn,"loadPretrained",0)
    # ------------------
    
    config  = get_config()
    print(" Create model ...")
    model = call_by_inspect( get_model, config)
    print(" ... Load weights ...")
    path_ckpt = Path("/home/yumeng/workspace/DDPM-SuJL/Results-old/ddpm2__v2/version_1/weights/model.weights").as_posix()
    model.load_weights(path_ckpt)
    # ------------------
    
    path_uncond = Path(log_dir,"uncond.png").as_posix()
    sample(model,path_uncond,4,None,0,img_size=config["img_size"], beta=config["beta"], bar_beta=config["bar_beta"], alpha=config["alpha"], sigma=config["sigma"], T=config["T"])
    path_uncond_ddim = Path(log_dir,"uncond_ddim.png")
    sample_ddim(model,path_uncond_ddim,4,None,1,1,img_size = config["img_size"],bar_alpha = config["bar_alpha"])