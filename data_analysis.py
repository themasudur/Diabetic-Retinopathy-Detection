import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import Counter

folder = 'D:\\Problem_1_Diabetic_Retinopathy_Detection_using_Color_Fundus_Photos\\ODIR_Data'

sample_data = np.load(os.path.join(folder, 'train', '1_left.npz'))

list(sample_data.keys())

sample_data['race'], sample_data['male'], sample_data['hispanic'], sample_data['maritalstatus'], sample_data['language']

sample_data['dr_class'], sample_data['dr_subtype']

sample_data['slo_fundus'].shape

def plot_2dmap(data, title=''):
    plt.imshow(data, cmap='viridis', aspect='auto')
    plt.colorbar(label='Thickness')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.
    plt.show()

plot_2dmap(sample_data['slo_fundus'])

sample_data1 = np.load(os.path.join(folder, 'train', '1_left.npz'))
sample_data2 = np.load(os.path.join(folder, 'train', '491_right.npz'))
sample_data3 = np.load(os.path.join(folder, 'train', '2523_right.npz'))
sample_data4 = np.load(os.path.join(folder, 'train', '3004_left.npz'))

plt.figure(figsize=(15, 4))

plt.subplot(1, 4, 1)
plt.imshow(sample_data['slo_fundus'])

plt.subplot(1, 4, 2)
plt.imshow(sample_data4['slo_fundus'])

plt.subplot(1, 4, 3)
plt.imshow(sample_data3['slo_fundus'])

plt.subplot(1, 4, 4)
plt.imshow(sample_data2['slo_fundus'])

plt.tight_layout()
plt.savefig("data.png")
plt.show()


meta = pd.read_csv(os.path.join(folder, 'data_summary.csv'))
meta.tail(10)

meta.info()

sex = Counter(meta['Patient Sex']).keys()
scounts = Counter(meta['Patient Sex']).values()

plt.bar(sex, counts)
plt.title("Patient Sex")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()