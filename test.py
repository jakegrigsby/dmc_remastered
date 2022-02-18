import dmc_remastered as dmcr

import matplotlib.pyplot as plt

env, _ = dmcr.visual_generalization("walker", "walk", 100000)
imgs = [env.reset() for _ in range(16)]

from mpl_toolkits.axes_grid1 import ImageGrid

fig = plt.figure(figsize=(5.0, 5.0))
grid = ImageGrid(fig, 111, nrows_ncols=(4, 4))

for ax, im in zip(grid, imgs):
    ax.imshow(im[:3].transpose(1, 2, 0))

plt.show()
