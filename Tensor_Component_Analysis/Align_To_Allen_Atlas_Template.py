import numpy as np
import matplotlib.pyplot as plt


allen_atlas_file = "/home/matthew/Documents/Allen_Atlas_Templates/Atlas_Template_V2.npy"

allen_atlas = np.load(allen_atlas_file)

plt.imshow(allen_atlas)
plt.show()