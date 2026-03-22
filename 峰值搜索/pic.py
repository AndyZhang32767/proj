import numpy as np
import matplotlib.pyplot as plt
from noise import snoise2 # 切换到 snoise2
import random

def generate_simplex_noise(width=512, height=512, filename="noise.png"):
    scale = 100.0
    octaves = 6
    
    # 随机种子：snoise2 对大种子值的鲁棒性更好
    seed = random.uniform(0, 10000)
    print(f"当前使用种子: {seed}")
    
    noise_image = np.zeros((height, width))
    
    for i in range(height):
        for j in range(width):
            # snoise2 参数与 pnoise2 略有不同，它没有 base 参数
            # 我们直接将种子作为坐标的偏移量（Z 轴偏移的思想）
            noise_image[i][j] = snoise2(i / scale + seed, 
                                        j / scale + seed, 
                                        octaves=octaves, 
                                        persistence=0.5, 
                                        lacunarity=2.0)

    # 归一化
    img_min, img_max = noise_image.min(), noise_image.max()
    noise_image = (noise_image - img_min) / (img_max - img_min) * 255
    noise_image = noise_image.astype(np.uint8)

    # 保存（覆盖写入）
    plt.imsave(filename, noise_image, cmap='gray')
    print(f"柏林噪声已保存至: {filename}")

generate_simplex_noise()