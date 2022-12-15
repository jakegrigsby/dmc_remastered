import sys
import dmc_remastered as dmcr

import matplotlib.pyplot as plt


if sys.argv[1] == "dynamics":
    env, _ = dmcr.dynamics_generalization(
        sys.argv[2], sys.argv[3], 100000, visual_seed=283
    )
elif sys.argv[1] == "visuals":
    env, _ = dmcr.visual_generalization(sys.argv[2], sys.argv[3], 100000)
elif sys.argv[1] == "both":
    env, _ = dmcr.full_generalization(sys.argv[2], sys.argv[3], 100000)
elif sys.argv[1] == "goals":
    env, _ = dmcr.goal_generalization(sys.argv[2], sys.argv[3], 10000)
else:
    raise ValueError()

imgs = []
for _ in range(16):
    env.reset()
    imgs.append(env.render_env())

from mpl_toolkits.axes_grid1 import ImageGrid

fig = plt.figure(figsize=(10.0, 10.0))
grid = ImageGrid(fig, 111, nrows_ncols=(4, 4))

for ax, im in zip(grid, imgs):
    ax.imshow(im)

plt.show()
