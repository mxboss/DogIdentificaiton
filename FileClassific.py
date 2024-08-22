import os
import shutil


def organize_files_by_prefix(target_folder):
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        print(f"目标文件夹 {target_folder} 不存在。")
        return

    # 遍历目标文件夹中的所有文件
    for filename in os.listdir(target_folder):
        # 确保只处理文件，忽略文件夹
        if os.path.isfile(os.path.join(target_folder, filename)):
            # 分割文件名前缀和后缀
            prefix = filename.split('_')[0]

            # 创建以前缀命名的文件夹
            prefix_folder = os.path.join(target_folder, prefix)
            if not os.path.exists(prefix_folder):
                os.makedirs(prefix_folder)

            # 移动文件到对应的前缀文件夹中
            shutil.move(os.path.join(target_folder, filename), os.path.join(prefix_folder, filename))

    print("文件整理完成。")


# 使用示例
target_folder = "./data/images"
organize_files_by_prefix(target_folder)
