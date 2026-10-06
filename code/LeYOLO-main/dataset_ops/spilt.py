import os
import shutil

# # 数据集文件夹路径
# dataset_folder = '/home/s0433/local/data/spark-2022-stream-1/images/train'  # 替换为你的数据集文件夹路径
# output_folders = [f'output_{i}' for i in range(1, 7)]  # 创建六个输出文件夹
#
# # 创建输出文件夹
# for folder in output_folders:
#     os.makedirs(folder, exist_ok=True)
#
# # 获取所有文件名
# files = sorted(os.listdir(dataset_folder))  # 按文件名排序
#
# # 按顺序分配到六个文件夹
# for i, file in enumerate(files):
#     target_folder = output_folders[i % 6]  # 循环分配到六个文件夹
#     shutil.copy(os.path.join(dataset_folder, file), os.path.join(target_folder, file))
#
# print("数据集已按顺序分配到六个文件夹中")
'''------------------------分割线----------------------------------'''
import os

# 定义文件夹路径和输出文件名
folder_path = '/home/s0433/local/projects_py/LeYOLO-main/dif'  # 替换为你的txt文件所在文件夹路径
output_file = '../../halfConv/data/SPARK-1/merged_file.txt'  # 输出文件名

# 获取文件夹中的所有txt文件
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# 按文件名排序（确保合并顺序）
txt_files.sort()

# 打开输出文件
with open(output_file, 'w') as outfile:
    # 遍历每个txt文件
    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)
        # 打开当前txt文件并读取内容
        with open(file_path, 'r') as infile:
            content = infile.read()
            outfile.write(content)  # 将内容写入输出文件
            outfile.write('\n')  # 在每个文件内容后添加换行符（可选）

print(f"所有txt文件已合并到 {output_file}")