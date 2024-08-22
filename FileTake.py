import os
import shutil


def move_files_to_root(target_folder):
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        print(f"目标文件夹 {target_folder} 不存在。")
        return

    # 遍历目标文件夹中的所有子文件夹
    for subdir in os.listdir(target_folder):
        subdir_path = os.path.join(target_folder, subdir)

        # 检查是否是子文件夹
        if os.path.isdir(subdir_path):
            # 遍历子文件夹中的所有文件
            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)

                # 移动文件到目标文件夹的根目录
                if os.path.isfile(file_path):
                    shutil.move(file_path, target_folder)

            # 删除空子文件夹
            os.rmdir(subdir_path)

    print("文件移动完成。")


# 使用示例
target_folder = "./data/images"
move_files_to_root(target_folder)
