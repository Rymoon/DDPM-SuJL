# Document of ddpm2.py

**Filetree**:
- ddpm2.py: training and sampling
- ddpm2_h.py: utils, not model
- ddpm2_m.py: model, optimizer, trainer/callback
- rennet: From MyHumbleADMMM7/RenNet/framework/Core; Things not depends on pytorch.
- bak_ddpm2.py.bak: Original copy from Jianlin SU.



## SubProcedure in `ddpm2.py`



### def get_config:

Image resized as H*W: img_size * img_size;
$$
\begin{aligned}
\text{T float}~&T &= 1000\\
\text{alpha vec}~&\alpha_t &= \sqrt{1-0.002t/T},~ t \in {1,2,\cdots,T}\\
\text{beta vec}~&\beta_t &= 1-\alpha_t^2\\
\text{bar\_alpha vec}~&\bar{\alpha}_t &= \prod_{s=1}^{s=T} \alpha_t\\
\text{bar\_beta vec}~&\bar{\beta}_t &=\sqrt{1-\bar{\alpha}_t^2}\\
\text{simga vec}~&\sigma_t &= \beta_t\\
\end{aligned}
$$


### def data_generator:

- batch_imgs:     N,H,W,C for RGB images;  
- batch_steps:    (batch_size, ) $\sim \mathcal{U}\{0,\cdots,T-1\}$  
- batch_noise:    N,H,W,C $\mathcal{n}(0,I)$;  

batch_noisy_imgs(a batch of $x_t$):
$$
x_t: = \bar{\alpha}_tx_{0} + \bar{\beta}_t \bar{\epsilon}_t;
$$ 

+ Shape of `bar_alpha` ?is list-T[float] ?
+ `bar_alpha[batch_steps][:, None, None, None]` means expand to 4D tensor?


### def sample:

Batch size is $n^2$, will output as an image grid (`figure`);

When $t0=0$, z_samples=None,
- `z_samples`(loop var): $y_t$
- `model.predict`(DNN): $\epsilon_\theta(y_t,t)$
0. $y_T:=\mathcal{N}(0,I)$ to HW3
1. $y_{t-1}:=\frac{1}{\alpha_t}\left(y_{t}-\frac{\beta_t^2}{\bar{\beta}_t} * \epsilon_\theta(y_{t},t)\right) + \sigma_t z$ , $z \sim \mathcal{N}(0,1)$ to HW3.
