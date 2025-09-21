import argparse
import cv2
import numpy as np
import torch
import os

# import os
# import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils_2 import GradCAM, show_cam_on_image, center_crop_img
# from vit_model import vit_base_patch16_224
import argparse
import torchvision.models as models
from model.TransGeo_20240711 import TransGeo

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--resume',
                    default='E:\\投稿CC\\visual\\checkpoint.pth.tar',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--dim', default=1000, type=int,
                    help='feature dimension (default: 128)')

parser.add_argument('--dataset', default='cvact', type=str,
                    help='vigor, cvusa, cvact')

parser.add_argument('--sat_res', default=320, type=int,
                    help='resolution for satellite')

parser.add_argument('--crop', default=True, action='store_true',
                    help='nonuniform crop')

parser.add_argument('--fov', default=0, type=int,
                    help='Fov')
args = parser.parse_args()
print(args)


class ReshapeTransform:
    def __init__(self, model):
        input_size = [256, 256]
        patch_size = [16, 16]
        self.h = input_size[0] // patch_size[0]
        self.w = input_size[1] // patch_size[1]

    def __call__(self, x):
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        result = x[:, 2:, :].reshape(x.size(0),
                                     self.h,
                                     self.w,
                                     x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result


model = TransGeo(args)
model = torch.nn.DataParallel(model).cuda()
# model = resnet50(pretrained=True)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
        if not args.crop:
            args.start_epoch = checkpoint['epoch']
        # best_acc1 = checkpoint['best_acc1']
        if args.crop and args.sat_res != 0:
            pos_embed_reshape = checkpoint['state_dict']['module.reference_net.pos_embed'][:, 2:, :].reshape(
                [1,
                 np.sqrt(checkpoint['state_dict']['module.reference_net.pos_embed'].shape[1] - 2).astype(int),
                 np.sqrt(checkpoint['state_dict']['module.reference_net.pos_embed'].shape[1] - 2).astype(int),
                 model.module.reference_net.embed_dim]).permute((0, 3, 1, 2))
            checkpoint['state_dict']['module.reference_net.pos_embed'] = \
                torch.cat([checkpoint['state_dict']['module.reference_net.pos_embed'][:, :2, :],
                           torch.nn.functional.interpolate(pos_embed_reshape, (
                               args.sat_res // model.module.reference_net.patch_embed.patch_size[0],
                               args.sat_res // model.module.reference_net.patch_embed.patch_size[1]),
                                                           mode='bilinear').permute((0, 2, 3, 1)).reshape(
                               [1, -1, model.module.reference_net.embed_dim])], dim=1)

        model.load_state_dict(checkpoint['state_dict'])
        # model.load_state_dict(checkpoint['state_dict'], strict=False)
        # if args.op == 'sam' and args.dataset != 'cvact':    # Loading the optimizer status gives better result on CVUSA, but not on CVACT.
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        # print("=> loaded checkpoint '{}' (epoch {})"
        #       .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

model_reference = model.module.reference_net
target_layers = [model_reference.blocks[-1].norm1]

data_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
# load image
img_path = "E:\\videodecode\\i2vgen-xl\\data\\Algonauts_2023\\output_image_320x320_6.jpg"
assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
img = Image.open(img_path).convert('RGB')
img = np.array(img, dtype=np.uint8)
img = center_crop_img(img, 320)
# [C, H, W]
img_tensor = data_transform(img)
# expand batch dimension
# [C, H, W] -> [N, C, H, W]
input_tensor = torch.unsqueeze(img_tensor, dim=0)
input_tensor = input_tensor.cuda()
reshape_transform = ReshapeTransform(model_reference)
cam = GradCAM(model=model_reference,
              target_layers=target_layers,
              use_cuda=False,
              reshape_transform=ReshapeTransform(model_reference))
target_category = 3  # tabby, tabby cat
# target_category = 254  # pug, pug-dog

grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(img / 255., grayscale_cam, use_rgb=True)
plt.imshow(visualization)
plt.show()

# model = torch.hub.load('facebookresearch/deit:main',
# 'deit_tiny_patch16_224', pretrained=True)
# model.eval()

# 判断是否使用 GPU 加速
# use_cuda = torch.cuda.is_available()
# if use_cuda:
#     model = model.cuda()


# def reshape_transform(tensor, height=20, width=20):
#     # 去掉类别标记
#     result = tensor[:, 2:, :].reshape(tensor.size(0),
#     height, width, tensor.size(2))

#     # 将通道维度放到第一个位置
#     result = result.transpose(2, 3).transpose(1, 2)
#     return result

# target_layers = [model.reference_net.norm]
# # 创建 GradCAM 对象
# cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda, reshape_transform=reshape_transform)

# # 读取输入图像
# image_path = "D:\\20240625\\test_8884\\reference\\__-DFIFxvZBCn1873qkqXA_satView_polish.jpg"
# rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
# rgb_img = cv2.resize(rgb_img, (320, 320))

# # 预处理图像
# input_tensor = preprocess_image(rgb_img,
# mean=[0.485, 0.456, 0.406],
# std=[0.229, 0.224, 0.225])

# # 将图像转换为批量形式
# input_tensor = input_tensor.unsqueeze(0)
# if use_cuda:
#     input_tensor = input_tensor.cuda()


# # target_layers = [model.reference_net.norm]#如果传入多个layer，cam输出结果将会取均值
# # 计算 grad-cam
# target_category = 100 # 可以指定一个类别，或者使用 None 表示最高概率的类别
# grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

# # 将 grad-cam 的输出叠加到原始图像上
# visualization = show_cam_on_image(rgb_img, grayscale_cam)

# # 保存可视化结果
# cv2.imwrite('cam.jpg', visualization)









