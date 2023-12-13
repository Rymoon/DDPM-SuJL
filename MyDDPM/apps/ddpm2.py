#! -*- coding: utf-8 -*-
# 生成扩散模型DDPM参考代码2
# 这版U-Net结构尽量保持跟原版一致（除了没加Attention），效果相对更好，计算量也更大
# 实验环境：tf 1.15 + keras 2.3.1 + bert4keras（当前Github最新版本，不能用pip安装的版本）
# 博客：https://kexue.fm/archives/9152

import os
gpuid = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpuid}"

from MyDDPM.apps.ddpm2_h import imwrite, list_pictures, imread, create_next_version_dir
from ren_utils.rennet import call_by_inspect, getitems_as_dict, root_Results
import ren_utils.rennet as rennet

from tqdm import tqdm
import numpy as np
import os
from pathlib import Path



def make_filetree(root_Results,*, train_name):
    """
    + logdir = root_Results/{train_name}/version_{number},
    + logdir/
        - samples
        - weights
    """
    p= Path(root_Results,train_name)
    p.mkdir(exist_ok=True,parents=True)
    logdir = create_next_version_dir(str(p))
    Path(logdir,"samples").mkdir(exist_ok=True,parents=True)
    Path(logdir,"weights").mkdir(exist_ok=True,parents=True)
    return logdir




def get_imgs(datasets =["CelebAHQ256_2"]):
    """nparry of imgs"""
    imgs = []
    for _name in datasets:
        imgs += list_pictures(rennet.datasets[_name]["imgs"],rennet.datasets[_name]["suffix"] )
    np.random.shuffle(imgs)
    return imgs

def data_generator(imgs,*, img_size, batch_size, T, bar_alpha, bar_beta):
    """图片读取
    """
    batch_imgs = []
    while True:
        for i in np.random.permutation(len(imgs)):
            batch_imgs.append(imread(imgs[i], img_size))
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


def sample_inter(model, path, n=4, k=8, sep=10, t0=500, *, imgs, img_size, beta, bar_beta, alpha , bar_alpha, T ):
    """随机采样插值函数
    
    Involved in `call_by_inspect`
    """
    
    x_samples = [imread(f) for f in np.random.choice(imgs, n * 2)]
    X = []
    for i in range(n):
        figure[i * img_size:(i + 1) * img_size, :img_size] = x_samples[2 * i]
        figure[i * img_size:(i + 1) * img_size,
               -img_size:] = x_samples[2 * i + 1]
        for j in range(k):
            lamb = 1. * j / (k - 1)
            x = x_samples[2 * i] * (1 - lamb) + x_samples[2 * i + 1] * lamb
            X.append(x)
    x_samples = np.array(X) * bar_alpha[t0]
    x_samples += np.random.randn(*x_samples.shape) * bar_beta[t0]
    x_rec_samples = call_by_inspect(sample, locals(), z_samples=x_samples) 
    if path is None:
        return x_rec_samples
    figure = np.ones((img_size * n, img_size * (k + 2) + sep * 2, 3))
    for i in range(n):
        for j in range(k):
            ij = i * k + j
            figure[i * img_size:(i + 1) * img_size, img_size * (j + 1) +
                   sep:img_size * (j + 2) + sep] = x_rec_samples[ij]
    imwrite(path, figure)

# Settings
def get_config(train_name:str):
    """基本配置, return as dict
    """
    
    img_size = 128  # 如果只想快速实验，可以改为64
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
    
    return dict(locals())

if __name__ == '__main__':
    DEBUG = False
    train_name = "ddpm2__v2" if not DEBUG else "ddpm2__v2__try"
    
    from MyDDPM.apps.ddpm2_m import get_model, Trainer
    config = get_config(train_name)
    
    logdir = make_filetree(root_Results,train_name=train_name)
    
    imgs =get_imgs(datasets = ["CelebAHQ256","CelebAHQ256_valid"]) if not DEBUG else get_imgs(datasets = ["CelebAHQ256_2"]) 
    model = call_by_inspect( get_model, config)
    
    # model.load_weights('model.ema.weights') # if want continue
    
    trainer = Trainer(model, model.optimizer, 
                    lambda model, path: sample(model, path, n=4, z_samples=None, t0=0, img_size=config["img_size"], beta=config["beta"], bar_beta=config["bar_beta"], alpha=config["alpha"], sigma=config["sigma"], T=config["T"]), 
                    logdir)
    model.fit(
        call_by_inspect(data_generator, config,  imgs=imgs),
        steps_per_epoch=30000, # ori:2000
        epochs=1000 if not DEBUG else 14,  # 只是预先设置足够多的epoch数，可以自行Ctrl+C中断
        callbacks=[trainer]
    )
    
    print("- logdir:",str(logdir))

