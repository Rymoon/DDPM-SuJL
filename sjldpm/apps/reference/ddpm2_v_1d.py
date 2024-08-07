"""
Use 1-dim data (a real number) instead of millions-dim data (image).
Short train, then draw histogram for many sampling. (To expose rare event).

Use FNN instead of Unet. Conv2D -> Conv1D.
"""


import os
gpuid = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpuid}"

from sjldpm.apps.reference.ddpm2_h import imwrite, create_next_version_dir
from sjldpm.apps.reference.rennet import call_by_inspect, getitems_as_dict, root_Results
import sjldpm.apps.reference.rennet as rennet

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


def get_config(train_name:str):
    """基本配置, return as dict
    """
    # tensor of NLC==batch_size, walker_dim, walker_ch
    walker_dim =1 
    walker_ch =1
    n_pat = 1000
    batch_size = 1000 # 10k floats
    embedding_size = 32
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

def get_vecs(n_pat):
    """
    n_pat different samples, 1,2,...,n_pat
    nparry of imgs: NLC
    Rescale to [-1,1]
    """
    data = (np.arange(n_pat)+1)/n_pat # uniform 
    data = data*2-1
    data = data.reshape(n_pat,1,1)
    np.random.shuffle(data)
    return data

def data_generator(vecs,*, walker_dim, batch_size, T, bar_alpha, bar_beta):
    """
    """
    while True:
        for i in range(round(len(vecs)/batch_size)+1):
            batch_indice = np.random.choice(T, len(vecs))
            batch_vecs = vecs[batch_indice]

            batch_steps = np.random.choice(T, batch_size)
            batch_bar_alpha = bar_alpha[batch_steps][:, None, None]
            batch_bar_beta = bar_beta[batch_steps][:, None, None]

            batch_noise = np.random.randn(*batch_vecs.shape)
            batch_noisy_imgs = batch_vecs * batch_bar_alpha + batch_noise * batch_bar_beta
            yield [batch_noisy_imgs, batch_steps[:, None]], batch_noise



def sample(model, path=None, n=1000, z_samples=None, t0=0, *, walker_dim, walker_ch, beta, bar_beta, alpha, sigma, T ):
    """随机采样函数.

    Sample many numbers, draw histogram and estimate the fake samples.
    """
    if z_samples is None:
        z_samples = np.random.randn(n, walker_dim, walker_ch)
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
    else:
        np.save(path,x_samples)



if __name__ == '__main__':
    DEBUG = True
    train_name = "ddpm2_v_1d" if not DEBUG else "ddpm2_v_1d"
    
    from sjldpm.apps.reference.ddpm2_m import get_model__vec, Trainer__vec
    config = get_config(train_name)
    
    logdir = make_filetree(root_Results,train_name=train_name)
    
    vecs = get_vecs(n_pat=config["n_pat"])
    model = call_by_inspect(get_model__vec, config)
    
    # model.load_weights('model.ema.weights') # if want continue
    
    trainer = Trainer__vec(model, model.optimizer, 
                    lambda model, path: call_by_inspect(sample, config, model=model, path=path, n=8*config["n_pat"], z_samples=None, t0=0), 
                    logdir)
    model.fit(
        call_by_inspect(data_generator, config,  vecs=vecs),
        steps_per_epoch=2000,
        epochs=5000 if not DEBUG else 14,  # 只是预先设置足够多的epoch数，可以自行Ctrl+C中断
        callbacks=[trainer]
    )
    
    print("- logdir:",str(logdir))

