import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def get_csv_files(directory):
    "获取制定目录下的所有csv文件"
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files

def read_csv_files_to_dict(csv_files):
    csv_dict = {}
    file_names = []
    for file_path in csv_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        file_names.append(file_name)
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            reader.fieldnames = [field.strip() for field in reader.fieldnames]
            csv_dict[file_name] = [row for row in reader]
    return csv_dict, file_names

def print_mAP50_95(csv_dict, file_names):
    column_name = 'metrics/mAP50-95(B)'.strip()
    def moving_average(interval, windowsize):
        p = windowsize // 2
        new_arr = np.concatenate((np.repeat(interval[0], p), interval, np.repeat(interval[-1], p)))
        window = np.ones(int(windowsize)) / float(windowsize)
        re = np.convolve(new_arr, window, "same")
        return re

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'lime','orange',
                       'purple', 'brown', 'pink',  'teal', 'navy', 'maroon'] # 颜色列表
    styles = ['--', '-', '-.', ':', '--', '-', '-.', ':', '--', '-', '-.', ':', '--', '-', '-.', ':', '--']
    num_colors = len(colors)

    for num, filename in enumerate(file_names):
        values = [float(row[column_name]) for row in csv_dict[filename] if column_name in row]
        if not values:
            print(f"No data found for {column_name} in {filename}. Skipping...")
            continue

        # print(f"Data for {filename}: {values}")
        color_index = num % num_colors  # 循环使用颜色
        style_index =color_index
        plt.xlim(00, len(values) - 1)
        plt.ylim(0.6, max(values) + 0.05)
        plt.plot(values, linestyle=styles[style_index], color=colors[color_index], label=filename)
        k = 7
        y_av = moving_average(values, k)
        sp_smooth = y_av[k//2:-(k//2)]
        # plt.plot(sp_smooth, color=colors[color_index])

    if file_names:
        plt.legend()
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('mAP50')
        plt.show()
    else:
        print("No data to plot.")

if __name__ == '__main__':
    csv_files = get_csv_files('results/')
    csv_dict, file_names = read_csv_files_to_dict(csv_files)
    print(f"CSV Files: {csv_files}")
    print(f"CSV Dict: {csv_dict.keys()}")
    print_mAP50_95(csv_dict, file_names)