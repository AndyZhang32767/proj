import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import label
import math # 用于计算欧几里得距离

def calculate_distance(p1, p2):
    #计算两点之间的欧几里得距离
    return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

def analyze_peaks_with_nms(image_path="noise.png", output_vis="check.png", nms_radius=20):
#    抑制半径。在此半径内的多个峰值将只保留最高的一个。

    # 1. 加载本地图片
    try:
        img_src = Image.open(image_path).convert('L')
        data = np.array(img_src)
    except FileNotFoundError:
        print(f"找不到文件 {image_path}")
        return

    # 2. 锁定阈值区间
    max_val = np.max(data)
    threshold = max_val - 30
    
    print("-" * 65)
    print(f"开始分析，抑制半径: {nms_radius}")
    print(f"全图最高灰度值: {max_val}, 筛选阈值 (Max-30): >={threshold}")
    print("-" * 65)

    # 3. 寻找初步连通区域
    mask = data >= threshold
    labeled_array, num_features = label(mask)
    
    initial_peaks = []

    # 4. 提取初步连通区域内的最高点
    for i in range(1, num_features + 1):
        region_indices = np.where(labeled_array == i)
        region_values = data[region_indices]
        
        local_max_idx = np.argmax(region_values)
        
        py, px = region_indices[0][local_max_idx], region_indices[1][local_max_idx]
        initial_peaks.append({'x': int(px), 'y': int(py), 'value': int(data[py, px])})

    print(f"初步检测到候选峰值: {len(initial_peaks)} 个")

    # 按灰度值从高到低排序
    initial_peaks.sort(key=lambda x: x['value'], reverse=True)

    # 5. 执行非极大值抑制 
    kept_peaks = []
    suppressed = np.zeros(len(initial_peaks), dtype=bool)

    for i in range(len(initial_peaks)):
        if suppressed[i]:
            continue 
        
        current_peak = initial_peaks[i]
        kept_peaks.append(current_peak) 
        
        for j in range(i + 1, len(initial_peaks)):
            if suppressed[j]:
                continue
            
            target_peak = initial_peaks[j]
            dist = calculate_distance(current_peak, target_peak)
            
            if dist < nms_radius:
                suppressed[j] = True

    # 再次排序方便输出
    kept_peaks.sort(key=lambda x: x['value'], reverse=True)

    # 6. 控制台详细输出
    print(f"抑制后最终保留主要峰值: {len(kept_peaks)} 个")
    print("-" * 50)
    print(f"{'序号':<5} | {'坐标 (X, Y)':<15} | {'灰度值':<10}")
    print("-" * 40)
    for idx, p in enumerate(kept_peaks, 1):
        print(f"{idx:<5} | ({p['x']:>4}, {p['y']:>4}) | {p['value']:>6}")
    print("-" * 65)

    # 7. 绘图并标注
    plt.figure(figsize=(12, 12))
    plt.imshow(data, cmap='gray')
    
    # 遍历最终保留的点
    for idx, p in enumerate(kept_peaks, 1):
        # --- 修改部分：统一使用红色 ---
        plt.scatter(p['x'], p['y'], s=300, edgecolors='red', facecolors='none', linewidths=2.5)
        
        # 标注文本，文字颜色也改为红色
        label_text = f"#{idx}:({p['x']},{p['y']})\nV:{p['value']}"
        plt.text(p['x'] + 10, p['y'] - 10, label_text, 
                 color='red', fontsize=9, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    plt.title(f"Peak Analysis with NMS (Radius={nms_radius}, {len(kept_peaks)} Final Nodes)")
    plt.axis('off')
    
    # 保存结果
    plt.savefig(output_vis, bbox_inches='tight', dpi=150)
    print(f"可视化结果已保存至: {output_vis}")
    plt.show()

# 执行
if __name__ == "__main__":
    # 如果发现还是扎堆，可以将 radius 调大，例如 50
    analyze_peaks_with_nms(nms_radius=30)