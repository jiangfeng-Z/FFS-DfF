"""
@author: zjf
@create time: 2024/1/3 16:15
@desc:
"""
import os

def check_and_create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 {folder_path} 不存在，已成功创建。")
    else:
        print(f"文件夹 {folder_path} 已存在。")