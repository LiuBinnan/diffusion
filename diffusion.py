import torch as th
import numpy as np

def beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return np.linspace(beta_start, beta_end, timesteps)

class Diffusion:

    def __init__(self, betas):

        self.betas = betas
        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas

        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        # q(x_{t-1}| x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef1 = (
            np.sqrt(self.alphas_cumprod_prev) * betas / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            np.sqrt(alphas) * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    # q(x_t| x_0)
    def q_mean_variance(self, x_start, t):
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        return mean, variance
    
    def q_sample(self, x_start, t, noise = None):
        if noise is None:
            noise = th.randn_like(x_start)
        return (
             _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
             _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    # q(x_{t-1}|x_t, x_0)
    def q_posterior_mean_variance(self, x_start, x_t, t):
        mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_start.shape) * x_start + 
            _extract_into_tensor(self.posterior_mean_coef2, t, x_start.shape) * x_t
        )
        variance = _extract_into_tensor(self.posterior_variance, t, x_start.shape)
        return mean, variance
    
    def p_mean_variance(self, model, x, t):

        # use model to predict the mean
        model_output = model(x, t)
        pred_xstart = self._predict_xstart_from_eps(x, t, model_output)
        model_mean, model_variance = self.q_posterior_mean_variance(pred_xstart, x, t)
        return model_mean, model_variance
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * (
                x_t - _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * eps
            )
        )
    
    def p_sample(self, model, x, t):
        mean, variance = self.p_mean_variance(model, x, t)
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0 
        return mean + nonzero_mask * th.sqrt(variance) * noise
    
    def p_sample_loop(self, model, shape, noise = None, progress = False):
        for sample in self.p_sample_loop_progressive(model, shape, noise, progress):
            final = sample
        return final
    
    def p_sample_loop_progressive(self, model, shape, noise = None, progress = False):
        device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                img = self.p_sample(model, img, t)
                yield img
    
    def ddim_sample(self, model, x, t, eta = 0.0):
        eps = model(x, t)
        x_start = self._predict_xstart_from_eps(x, t, eps)
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = eta * (
            th.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar)) *
            th.sqrt(1.0 - alpha_bar / alpha_bar_prev)
        )
        mean = (th.sqrt(alpha_bar_prev) * x_start + 
                th.sqrt(1.0 - alpha_bar_prev - sigma ** 2) * eps)
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        return mean + nonzero_mask * sigma * noise
    
    # eta = 0.0 for ddim inversion
    def ddim_reverse_sample(self, model, x, t):
        eps = model(x, t)
        x_start = self._predict_xstart_from_eps(x, t, eps)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)
        mean = (th.sqrt(alpha_bar_next) * x_start + 
                th.sqrt(1.0 - alpha_bar_next) * eps)
        return mean

    def ddim_sample_loop(self, model, shape, noise = None, progress = False, eta = 0.0):
        for sample in self.ddim_sample_loop_progressive(model, shape, noise, progress, eta):
            final = sample
        return final

    def ddim_sample_loop_progressive(self, model, shape, noise = None, progress = False, eta = 0.0):
        device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                img = self.ddim_sample(model, img, t, eta)
                yield img



def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

# arr = np.linspace(1, 5, 5)
# t = th.linspace(1, 3, 3, dtype=int)
# print(th.from_numpy(arr)[t][..., None, None].expand((3, 7, 7)))
