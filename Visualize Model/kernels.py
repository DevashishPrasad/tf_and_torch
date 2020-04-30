from torchvision.utils import make_grid
import matplotlib.pyplot as plt

kernels = model.layer1[0].weight.detach().clone()
kernels = kernels.cpu()
kernels = kernels - kernels.min()
kernels = kernels / kernels.max()
img = make_grid(kernels)
plt.imshow(img.permute(1, 2, 0))
