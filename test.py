from unet_ import UNet
from diffusion import Diffusion, beta_schedule
import torch as th
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from diffusers import DDPMPipeline

# image_pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
# image_pipe.to("cuda")
# images = image_pipe().images
# plt.imshow(images[0])
# plt.show()

# channels = 3
# device = th.device('cuda' if th.cuda.is_available() else 'cpu')
# model = UNet(image_channels=channels).to(device)
# model.load_state_dict(th.load("C:\\Users\\11326\\Desktop\\dl\\ddpm\\UNET.pt"))
# shape = (16, 3, 32, 32)
# betas = beta_schedule(1000)
# diff = Diffusion(betas)
# imgs = diff.ddim_sample_loop(model, shape, progress=True)
# reverse_transform = transforms.Compose([
#     transforms.Lambda(lambda t: (t + 1) / 2),
#     transforms.Lambda(lambda t: np.transpose(t, (0, 2, 3, 1))), # BCHW to BHWC
#     transforms.Lambda(lambda t: t * 255.),
#     transforms.Lambda(lambda t: t.astype(np.uint8)),
# ])
# imgs = reverse_transform(imgs.cpu().numpy())
# np.save("image.npy",imgs)
# imgs = np.load("image.npy")

# # 设置子图的行数和列数
# rows = 4
# cols = 4

# # 创建一个新的图形，设置水平和垂直间距为零
# fig, axs = plt.subplots(rows, cols, figsize=(8, 8), subplot_kw={'xticks': [], 'yticks': []}, gridspec_kw=dict(wspace=0, hspace=0))

# # 循环遍历图片并展示在子图中
# for i in range(rows):
#     for j in range(cols):
#         index = i * cols + j
#         axs[i, j].imshow(imgs[index])
#         axs[i, j].axis('off')  # 关闭坐标轴

# plt.show()