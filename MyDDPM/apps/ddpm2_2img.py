#! -*- coding: utf-8 -*-
# 生成扩散模型DDPM参考代码2
# 这版U-Net结构尽量保持跟原版一致（除了没加Attention），效果相对更好，计算量也更大
# 实验环境：tf 1.15 + keras 2.3.1 + bert4keras（当前Github最新版本，不能用pip安装的版本）
# 博客：https://kexue.fm/archives/9152
# Train on ONLY 1 image;

import os
gpuid = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpuid}"
train_name = "ddpm2_2img"

from ddpm2_h import * 

if not os.path.exists(f"{train_name}__samples"):
    os.mkdir(f"{train_name}__samples")

# 基本配置
_imgs = list_pictures('/home/yumeng/workspace/Dataset/CelebAHQ_2/', 'jpg')
# repeat up to 10k
n = len(_imgs)
imgs = []
for i in range(round(10000/n)+1):
    imgs+=_imgs 
np.random.shuffle(imgs)
img_size = 128  # 如果只想快速实验，可以改为64
batch_size = 16  # 如果显存不够，可以降低为16，但不建议低于16
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



def sample(model, path=None, n=4, z_samples=None, t0=0):
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


def sample_inter(path, n=4, k=8, sep=10, t0=500):
    """随机采样插值函数
    """
    figure = np.ones((img_size * n, img_size * (k + 2) + sep * 2, 3))
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
    x_rec_samples = sample(z_samples=x_samples, t0=t0)
    for i in range(n):
        for j in range(k):
            ij = i * k + j
            figure[i * img_size:(i + 1) * img_size, img_size * (j + 1) +
                   sep:img_size * (j + 2) + sep] = x_rec_samples[ij]
    imwrite(path, figure)


class Trainer(Callback):
    """训练回调器
    """
    def __init__(self,model,optimizer):
        self.model = model
        self.optimizer = optimizer
    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(f"{train_name}.weights")
        sample(f"{train_name}__samples/{(epoch + 1):05d}.png")
        self.optimizer.apply_ema_weights()
        self.model.save_weights(f"{train_name}.ema.weights")
        sample(f"{train_name}__samples/{(epoch + 1):05d}_ema.png")
        self.optimizer.reset_old_weights()

# TODO 
if __name__ == '__main__':

    model = get_model()
    trainer = Trainer(model,optimizer)
    model.fit(
        data_generator(),
        steps_per_epoch=2000,
        epochs=10000,  # 只是预先设置足够多的epoch数，可以自行Ctrl+C中断
        callbacks=[trainer]
    )

else:

    model.load_weights(f"{train_name}.ema.weights")
