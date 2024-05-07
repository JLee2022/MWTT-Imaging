# 包含绘图，可视化函数

import matplotlib.pyplot as plt
import numpy as np

def _mapminmax(data):
    d = data - data.min()
    return d / d.max()

def plot_hrrp(data, is_db, is_map):
    if is_db:
        data = 10 * np.log10(np.abs(data) + 1e-6)
    else:
        data = np.abs(data)

    # map min max
    if is_map:
        data = _mapminmax(data)
    else:
        pass

    plt.imshow(data, cmap="jet")
    plt.title('hrrp')
    plt.show(block=True)
    return None


def plot_img(data, is_db=True, is_map=True):
    if is_db:
        data = 10 * np.log10(np.abs(data) + 1e-6)
    else:
        data = np.abs(data)

    # map min max
    if is_map:
        data = _mapminmax(data)
    else:
        pass

    plt.imshow(data, cmap="jet")
    plt.title('imaging')
    plt.show(block=True)
    return None

