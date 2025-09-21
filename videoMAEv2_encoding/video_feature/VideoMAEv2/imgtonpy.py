# from PIL import Image
# import numpy as np
#
# a=np.load("E:/videodecode/i2vgen-xl/data/Algonauts_2023/npy/0001.npy")
#
# def jpg_to_npy(jpg_path, npy_path):
#     # 打开JPEG图像文件
#     img = Image.open(jpg_path)
#
#     # 将图像转换为NumPy数组
#     img_array = np.array(img)
#
#     # 将NumPy数组保存为.npy文件
#     np.save(npy_path, img_array)
#
#
# # 示例使用
# jpg_path = 'E:/videodecode/i2vgen-xl/data/Algonauts_2023/images/0001.jpg'  # 替换为你的JPEG图像文件路径
# npy_path = 'E:/videodecode/i2vgen-xl/data/Algonauts_2023/npy/0001.npy'  # 替换为你想保存.npy文件的路径
#
# jpg_to_npy(jpg_path, npy_path)


from PIL import Image
import numpy as np
import os


def jpg_to_npy_batch(jpg_folder, npy_folder):
    # Ensure the output folder exists
    os.makedirs(npy_folder, exist_ok=True)

    # List all JPEG files in the input folder
    jpg_files = os.listdir(jpg_folder)

    for jpg_file in jpg_files:
        # Construct full paths
        jpg_path = os.path.join(jpg_folder, jpg_file)
        npy_path = os.path.join(npy_folder, os.path.splitext(jpg_file)[0] + '.npy')

        try:
            # Open JPEG image
            img = Image.open(jpg_path)

            # Convert image to NumPy array
            img_array = np.array(img)

            # Save NumPy array as .npy file
            np.save(npy_path, img_array)
            print(f"Converted {jpg_file} to {os.path.basename(npy_path)}")

        except IOError:
            print(f"Error converting {jpg_file}")


# Example usage
jpg_folder = 'E:/videodecode/i2vgen-xl/data/Algonauts_2023/images'  # Replace with your JPEG images folder path
npy_folder = 'E:/videodecode/i2vgen-xl/data/Algonauts_2023/npy'  # Replace with your output folder path

jpg_to_npy_batch(jpg_folder, npy_folder)












