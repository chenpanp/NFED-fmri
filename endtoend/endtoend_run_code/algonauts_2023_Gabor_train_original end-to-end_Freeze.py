


import CLIP.clip as clip
import pickle
# import numpy as np
# ### 0.1.3 Import the libraries
# aa=np.load("/media/amax/DE10576510574425/endtoend/data_mean/sub05/V3B.npy")
#V2 4731 V3 4052 V4 1625 FFA 4222 IPS 7877 LO 2257 OFA 1611 PSTS 2385 TPJ 5431 V3A 1581 V3B 939
from sklearn.decomposition import PCA
import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn import plotting
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr as corr


#from SFCNet import *

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from GallantData import *
from GaborLayers import *
import scipy.io as io
from optparse import OptionParser
import numpy as np
import visdom
import cv2
import os
import sys
import math
from torchvision import transforms


from VideoMAEv2.extract_tad_feature import get_args as get_args_video
from VideoMAEv2.models import vit_giant_patch14_224
from VideoMAEv2.extract_tad_feature import extract_feature
from VideoMAEv2.extract_tad_feature import get_video_loader
from VideoMAEv2.extract_tad_feature import get_start_idx_range
from VideoMAEv2.extract_tad_feature import ToFloatTensorInZeroOne
from VideoMAEv2.extract_tad_feature import Resize
from VideoMAEv2.models import vit_small_patch16_224



# clip_model, preprocess = clip.load("ViT-B/32", device=device)


class argObj:
    def __init__(self, data_dir, parent_submission_dir, subj):
        self.subj = format(subj, '02')
        self.data_dir = os.path.join(data_dir, 'subj' + self.subj)
        self.parent_submission_dir = parent_submission_dir
        self.subject_submission_dir = os.path.join(self.parent_submission_dir,
                                                   'subj' + self.subj)


        if not os.path.isdir(self.subject_submission_dir):
            os.makedirs(self.subject_submission_dir)



class SFCLinear(nn.Module):
    def __init__(self, in_features, out_features, c, h, w, c1):
        super(SFCLinear, self).__init__()

        self.w = nn.Parameter(torch.randn(in_features, out_features))
        self.b = nn.Parameter(torch.randn(out_features))

        self.w.data.uniform_(-0.1, 0.1)
        self.b.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        x = x.mm(self.w ** 2)
        return x + self.b


class FC_direct(nn.Module):
    def __init__(self, num):
        super(FC_direct, self).__init__()

        self.w = nn.Parameter(torch.randn(num))
        self.b = nn.Parameter(torch.randn(num))

    def forward(self, x):
        x = x * self.w + self.b
        return x

class ConvBlock(nn.Module):  ## 卷积模块？ GABOR
    def __init__(self, inchannel, outchannel, kernel, stride):
        super(ConvBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=kernel, stride=stride, padding=(kernel - 1) // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.layer(x)
        return out





class GaborNetDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_paths, anno_trn_path, idxs, vs, transform,video_paths):
        self.transform = transform
        self.imgs_paths = np.array(imgs_paths)[idxs]
        self.voxels = vs[idxs]
        self.video_paths = video_paths



        with open(anno_trn_path, 'rb') as f:
            self.annotations = pickle.load(f)  # 加载 pickle 文件

    def __getitem__(self, idx):
        img_path = self.imgs_paths[idx]


        img_name = os.path.basename(img_path)


        nsd_id = img_name.split('_')[1].rsplit('.', 1)[0]
        if nsd_id == '':
            nsd_id = '0'
        nsd_id = int(nsd_id)

        v = self.voxels[idx]

        annotation = self.annotations[nsd_id]  # 获取对应的注释,这时候type是list

        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img).to(device)

        video_loader = get_video_loader()
        start_idx_range = get_start_idx_range("THUMOS14")
        transform = transforms.Compose(
            [ToFloatTensorInZeroOne(),
             Resize((224, 224))])

        video_path = self.video_paths[idx]
        video_path = str(video_path)
        vr = video_loader(video_path)



        feature_list = []
        num_frames = 16
        for i in range(num_frames):
            start_idx = i * (len(vr) // num_frames)
            data = vr.get_batch(np.arange(start_idx, start_idx + 1)).asnumpy()
            frame = torch.from_numpy(data)  # torch.Size([16, 566, 320, 3])
            frame_q = transform(frame)  # torch.Size([3, 16, 224, 224])
            input_data = frame_q.unsqueeze(0).cuda()
            feature_list.append(input_data)
        video = torch.cat(feature_list, dim=0).squeeze(1).squeeze(2).permute(1,0,2,3)




        return img, v, annotation,video

    def __len__(self):
        return len(self.imgs_paths)



class SFCNet(nn.Module):  ##   Garbor卷积和，虚实两条，各64通道，  大小9*9   gabor
    def __init__(self, size, base_channel=128, block_num=2, kernel=3, stride=2, num=1294, selected_channel=4):
        super(SFCNet, self).__init__()
        self.inchannel = base_channel

        # self.conv0 = GaborConv2d0(3, 4, kernel_size=9, stride=2, padding=2)
        # self.conv1 = GaborConv2d1(3, 4, kernel_size=9, stride=2, padding=2)

        self.relu = nn.ReLU(inplace=True)
        # self.layer = self.make_layer(ConvBlock, base_channel, block_num, kernel, stride)
        #
        # h = size // (stride ** (block_num + 1))
        # w = size // (stride ** (block_num + 1))
        # out_dims = h * w * self.inchannel

        self.fc = nn.Linear(1408, 4222)
        # 6404 V2 4731 V3 4052 V4 1625 FFA 4222 IPS 7877 LO 2257 OFA 1611 PSTS 2385 TPJ 5431 V3A 1581 V3B 939  MT 2147

        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)

        for param in self.clip_model.parameters():
            param.requires_grad = False


        self.model_Video = vit_giant_patch14_224(
            img_size=224,
            pretrained=False,
            num_classes=710,
            all_frames=16,
            tubelet_size=2,
            drop_path_rate=0.3,
            use_mean_pooling=True)

        ckpt = torch.load("/home/amax/endtoend/GaborNet-zhao/VideoMAEv2/checkpoints/vit_g_hybrid_pt_1200e_k710_ft.pth",
                          map_location='cpu')
        # ckpt = torch.load("/home/amax/endtoend/GaborNet-zhao/VideoMAEv2/checkpoints/vit_s_k710_dl_from_giant.pth",
        #                   map_location='cpu')
        for model_key in ['model', 'module']:
            if model_key in ckpt:
                ckpt = ckpt[model_key]
                break
        self.model_Video.load_state_dict(ckpt)
        # self.model_Video.eval()
        self.model_Video.cuda()

        for name, param in self.model_Video.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name} | Size: {param.numel()}")

        # 统计总参数数量
        total_params = sum(p.numel() for p in self.model_Video.parameters() if p.requires_grad)
        print(f"Total Trainable Parameters: {total_params}")




    def make_layer(self, block, channels, num_blocks, kernel, stride):
        layers = []
        for i in range(num_blocks):
            if i == (num_blocks - 1):
                layers.append(block(self.inchannel, channels, kernel, stride))
            else:
                layers.append(block(self.inchannel, channels, kernel, stride))
            self.inchannel = channels
            channels = channels * 1

        return nn.Sequential(*layers)





    def extract_video_features(self,video):
        with torch.no_grad():
          video_feature =self.model_Video.forward_features(video).to(device)
        return video_feature

    def extract_clip_features(self, x, annotation):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(x)
            batch_size = image_features.size(0)


            captions = clip.tokenize(annotation, truncate=True)
            caption_features = self.clip_model.encode_text(captions.to(device))


            image_features /= image_features.norm(dim=-1, keepdim=True)
            caption_features /= caption_features.norm(dim=-1, keepdim=True)




            image_features = torch.cat([image_features, caption_features], dim=1)

            return image_features, caption_features




    def forward(self, x, annotation,video):
        clip_image_features, clip_anno_features = self.extract_clip_features(x, annotation)

        video_features=self.extract_video_features(video)

        y = self.relu(video_features)
        y = y.float()
        y = self.fc(y)


        return  y




    def features(self, x):
        x = self.relu(self.conv1(x))
        x = self.layer(x)
        return x

    def Gabor_features(self, x):
        x = torch.cat((self.conv0(x), self.conv1(x)), dim=1)
        return x

    def top_pro(self, x):
        x = self.layer(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


def cor_v(x1, x2):   #gabor
    x1_mean, x1_var = torch.mean(x1, 0), torch.var(x1, 0)   #  0,按列求平均值,方差
    x2_mean, x2_var = torch.mean(x2, 0), torch.var(x2, 0)
    corVector = torch.mean((x1-x1_mean)*(x2-x2_mean), 0)
    corVector = corVector/(1e-6+torch.sqrt(x1_var*x2_var))   #
    return corVector


def loss_w(x1, x2):         #gabor
    corVector = cor_v(x1, x2)
#    loss1 = torch.mean(corVector)
    loss2 = torch.mean(corVector**3)
    return -loss2


def mse_v(x1, x2):          #gabor
    mseVector = torch.mean((x1-x2)**2, 0)
    return mseVector


def topk(x, k):             #gabor
    corTopkv, corTopki = torch.topk(x, k)
    corTopk = torch.mean(corTopkv)
    return corTopk, corTopki


def ptopk(x, ix):           #gabor
    numsv = ix.shape[0]
    tmp = 0
    for i in range(numsv):
        tmp = tmp + x[ix[i]]
    tmp = tmp/int(numsv)
    return tmp


## modify!!!!!!!
def eval_model_challenge(net, dataset, bs):  # （稀疏全连接网络，训练数据，70，300）
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0, drop_last=False)

    cor = 0.0
    mse = 0.0
    count = 0

    net.eval()

    all_voxel_vector = []

    for i, data in enumerate(loader):
        imgs, voxels, annotation,video = data  # 图像，标签，体素信号

        imgs = imgs.float().cuda()
        voxels = voxels.float().cuda()

        pred = net(imgs, annotation,video)

        cor_v_ = cor_v(pred, voxels)
        mse_v_ = mse_v(pred, voxels)
        cor_v_cpu = cor_v_.cpu().detach().numpy()
        sorted_cor_v = np.sort(cor_v_cpu)[::-1]

        cor = sorted_cor_v[:500]
        cor = cor[np.newaxis, :]

        # cor = cor + cor_v_.mean().item()

        mse = mse + mse_v_.mean().item()
        # all_voxel_vector[count, :] = cor
        all_voxel_vector.append(cor)

        count = count + 1

        # cor_v_cpu = cor_v_.cpu().detach().numpy()
        #
        # sorted_cor_v = np.sort(cor_v_cpu)[::-1]
        # top_100_cor_v = sorted_cor_v[:500]
        # average_top_100 = np.median(top_100_cor_v )
    all_voxel_vector = np.vstack(all_voxel_vector)
    cor_median = np.mean(all_voxel_vector, axis=0)
    cor = np.mean(cor_median)

    print(cor)
    # cor = cor/count
    mse = mse / count

    # all_voxel_vector = np.reshape(np.array(all_voxel_vector), (-1,))

    median = np.median(cor_median)

    return cor, mse, median



def eval_model(net, dataset, bs, k, index_cor=None, index_mse=None):  # （稀疏全连接网络，训练数据，70，300）
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0, drop_last=False)

    cor = 0.0
    mse = 0.0
    corVector = 0
    mseVector = 0
    count = 0

    net.eval()
    pred_eval = np.zeros((1, length_voxel))
    pred_eval = torch.tensor(pred_eval)
    pred_eval = pred_eval.cpu().numpy()
    for i, data in enumerate(loader):
        imgs, voxels = data  # 图像，标签，体素信号
        imgs = imgs.float().cuda()
        voxels = voxels.float().cuda()


        pred = net(imgs)
        cor_v_ = cor_v(pred, voxels)
        mse_v_ = mse_v(pred, voxels)

        cor = cor + cor_v_.mean().item()  # 这是一个数
        mse = mse + mse_v_.mean().item()  # 这是一个数

        corVector = corVector + cor_v_.data  # shape为(N,)
        mseVector = mseVector + mse_v_.data  # shape为(N,)

        count = count + 1


        pred = torch.tensor(pred)
        pred = pred.cpu().numpy()
        pred_eval = np.append(pred_eval, pred, axis = 0)


    cor = cor/count  # 这是所有体素取平均
    mse = mse/count  # 这是所有体素取平均
    corVector = corVector/count  # 这是每个体素自己取平均，shape为(N,)
    mseVector = mseVector/count  # 这是每个体素自己取平均，shape为(N,)

    if index_cor is not None:  # 默认为None
        cor_topk = ptopk(corVector, index_cor)
        mse_topk = ptopk(mseVector, index_cor)
    else:
        cor_topk, index_cor = topk(corVector, k)   # 均值，取前300个皮尔逊相关系数最大的体素的索引值(值越大越好)
        mse_topk, index_mse = topk(-mseVector, k)  # 均值，取前300个均方误差最小的体素的索引值
        mse_topk = -1*mse_topk  # (值越小越好)

    corVector = corVector.cpu().numpy()
    mseVector = mseVector.cpu().numpy()

    return corVector, cor, cor_topk, index_cor, mseVector, mse, mse_topk, index_mse, pred_eval  # 返回每个体素自己取平均，所有体素取平均，前300个体素取均值，前300个体素的索引值



def train_net(net,          # 稀疏全连接网络
              epochs,       # 50
              batchsize,    # 64
              lr,           # 0.001
              dataset_trn,  # 训练数据
              dataset_val, # 测试数据
              #dataset_test,
              k,            # 300，挑前300个体素进行相关性分析
              num,          # 体素数量
              viz,          # 可视化的啥玩意
              area,         # v1
              sub,
              save_cp=True,  # True
              gpu=True):     # True

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
        selected voxel num: {}
    '''.format(epochs, batchsize, lr, len(dataset_trn),
               len(dataset_val), str(save_cp), str(gpu), str(k)))

    trace = dict(title=title,
                 xlabel='epoch',
                 ylabel='correlation',
                 legend=['train', 'train_topk', 'test', 'test_topk'],
                 markersymbol='dot')

    trace_mse = dict(title=title_mse,
                     xlabel='epoch',
                     ylabel='mse',
                     legend=['train', 'train_topk', 'test', 'test_topk'],
                     markersymbol='dot')

    loader_trn = torch.utils.data.DataLoader(dataset_trn, batch_size=batchsize, shuffle=True, num_workers=0,
                                             drop_last=True)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)

    best_cor = 0.0
    best_median = 0.0
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))

        net.train()
        cor_trn = 0
        count = 0
        for i, data in tqdm(enumerate(loader_trn)):

            imgs, voxels, annotation, video= data    # 图像，标签，体素信号
            # print('annotation len = ', len(annotation))
            imgs = imgs.float().cuda()
            voxels = voxels.float().cuda()

            pred_v = net(imgs, annotation,video)  # 预测得到的大脑信号？
            pred_v = pred_v# + 1.0 * torch.randn(pred_v.shape).cuda()  # 加入标准正态分布的高斯白噪声

            loss = loss_w(pred_v, voxels) + 1e-5 * torch.mean(pred_v)  # 添加噪声正则化以防止预测体素V不断变大loss = loss_w(pred_v, voxels) + 1e-5 * torch.mean(pred_v)

            # cor
            cor_trn = cor_trn + cor_v(pred_v, voxels).mean().item()
            count = count + 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        cor_trn = cor_trn / count

        # cor_trn, mse_trn, median_trn = eval_model_challenge(net, dataset_trn, 4)
        cor_val, mse_val, median_val = eval_model_challenge(net, dataset_val, 4)


        print('\tloss:%.8f' % (loss.cpu().item()))  # 一个epoch结束之后的loss
        print('\ttrain cor:%.8f, val cor:%.8f' % (cor_trn, cor_val))  # 训练集所有体素的平均相关系数，测试集所有体素的平均相关系数print('\ttrain cor:%.3f, val cor:%.3f' % (cor_trn, cor_val))
        print('\tval median:%.8f' % (median_val))
        print('========END========')



        if cor_val > best_median:
            best_median = cor_val
            torch.save(net.state_dict(),r'/media/amax/DE105765105744252/endtoend/2023Challenge/duandaoduan_W/V2_endtoend.pth')  # 存储模型参数
            torch.save(net, r'/media/amax/DE105765105744252/endtoend/Results/改_1/sub01/model' + '/' + area + '_C_best_net.pth')  # 存储模型参数
            print('Checkpoint {} saved!'.format(epoch + 1))
            file = r'/media/amax/DE105765105744252/endtoend/Results/改_1/sub01/MED' + '/' + area + '_medVector_val.npy'
            np.save(file, median_val)  # 存储视觉区训练集各个体素的相关系数
            file = r'/media/amax/DE105765105744252/endtoend/Results/改_1/sub01/COR' + '/' + area + '_corVector_val.npy'
            np.save(file, cor_val)  # 存储视觉区训练集各个体素的相关系数



def get_args():
    parser = OptionParser()  # 构造optionparser的对象
    parser.add_option('-e', '--epochs', dest='epochs', default=20, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=4,
                      type='int', help='batch size')
    parser.add_option('-n', '--num', dest='num', default=math.ceil(num_v/2),
                      type='int', help='voxel number')
    parser.add_option('-l', '--learning-rate', dest='lr', default=1e-3,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    #parser.add_option('--data_dir', type=str, default='E:/match/')
    parser.add_option('--gpu_num', type=str, default='0')
    (options, args) = parser.parse_args()  # 调用解析函数

    return options


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

# #
    subj = 1  # @param ["1", "2", "3", "4", "5", "6", "7", "8"] {type:"raw", allow-input: true}
    sub_str = str(subj)

# ## 0.2 Access the tutorial data

    data_dir = r'/media/amax/DE105765105744252/endtoend/2023Challenge/data_code/algonauts_2023_challenge_data'
    parent_submission_dir = r'/media/amax/DE105765105744252/endtoend/2023Challenge/data_code/algonauts_2023_challenge_data/submission'

    anno_path = ''
    # load annotation
    anno_path = os.path.join(r'/media/amax/DE105765105744252/endtoend/2023Challenge/CLIP/anno', 'subj0' + format(subj, '1'))
    anno_trn_path = anno_path + '_trn_anno.pickle'
    # print('anno_trn_path', anno_trn_path)
    anno_tst_path = anno_path + '_tst_anno.pickle'

# ##
    device = 'cuda'  # @param ['cpu', 'cuda'] {allow-input: true}
    device = torch.device(device)

    # ## 1.2 Load the fMRI training data
    args = argObj(data_dir, parent_submission_dir, subj)
    fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')
    lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
    rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))


    train_img_dir = os.path.join(args.data_dir, 'training_split', 'training_images')
    train_video_dir=os.path.join(args.data_dir, 'training_split', 'training_videos')

    # print('train_img_dir', train_img_dir)
    test_img_dir = os.path.join(args.data_dir, 'test_split', 'test_images')

    # Create lists will all training and test image file names, sorted
    train_img_list = os.listdir(train_img_dir)
    train_img_list.sort()
    train_video_list = os.listdir(train_video_dir)
    train_video_list.sort()
    test_img_list = os.listdir(test_img_dir)
    test_img_list.sort()



    train_img_file = train_img_list[0]
    train_video_file = train_video_list[0]
    print('Training image file name: ' + train_img_file)
    print('73k NSD images ID: ' + train_img_file[-9:-4])



    img = 0  # @param
    video=0
    hemisphere = ['l', 'r']  # @param ['left', 'right'] {allow-input: true}
    print(hemisphere[0])
    print(hemisphere[1])
    # Load the image
    img_dir = os.path.join(train_img_dir, train_img_list[img])
    train_img = Image.open(img_dir).convert('RGB')
    rand_seed = 5  # @param
    np.random.seed(rand_seed)
    #### Load the video
    video_path_1 = os.path.join(train_video_dir,  train_video_list[video])
    cap = cv2.VideoCapture(video_path_1)

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB format
        frames.append(frame)

    cap.release()




    idxs = np.arange(len(train_video_list))
    idxs_train, idxs_val = idxs[:1200], idxs[1200:]


    idxs_test = np.arange(len(test_img_list))



    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # resize the images to 224x24 pixels
        transforms.ToTensor(),  # convert the images to a PyTorch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize the images color channels
    ])

    batch_size = 4  # @param 原始参数300
    # Get the paths of all image files
    train_imgs_paths = sorted(list(Path(train_img_dir).iterdir()))

    train_videos_paths = sorted(list(Path(train_video_dir).iterdir()))
    test_imgs_paths = sorted(list(Path(test_img_dir).iterdir()))

    rois = ["V1"]

    for i in range(len(rois)):
        roi = rois[i]

        print(roi)


        lh_fmri_roi =  np.load("/media/amax/DE105765105744252/endtoend/2023Challenge/data_code/algonauts_2023_challenge_data/subj01/training_split/training_fmri/lh_training_fmri.npy")

        GaborNetDataset_trn_L = GaborNetDataset(train_imgs_paths, anno_trn_path, idxs_train, lh_fmri_roi, transform,train_videos_paths)
        GaborNetDataset_val_L = GaborNetDataset(train_imgs_paths, anno_trn_path, idxs_val, lh_fmri_roi, transform,train_videos_paths)

        lh_fmri_val = lh_fmri_roi[idxs_val]

        try:
            length_voxel = len(lh_fmri_roi[0])
        except TypeError:
            length_voxel = False

        clip_model, preprocess = clip.load("ViT-B/32", device=device)

        size = 224  # 调整的是输入图像的分辨率9
        base_channel = 128
        block_num = 2
        kernel = 3
        selected_channel = 128
        stride = 2
        num_v = len(lh_fmri_roi[0])
        print('体素数 = ', num_v)
        net = SFCNet(size, base_channel, block_num, kernel, stride, num_v, selected_channel)  # (128,128,2,3,2,体素数量,128)

        print(net.parameters)  # 显示参数量大小

        #   可视化
        title = 'cor/ROI=%s/c=%d/sc=%d/bn=%d' % (
        roi, base_channel, selected_channel, block_num)  # cor/ROI=v1/c=128/sc=128/bn=2
        title_mse = 'mse/ROI=%s/c=%d/sc=%d/bn=%d' % (
        roi, base_channel, selected_channel, block_num)  # mse/ROI=v1/c=128/sc=128/bn=2
        viz_name = 'E_' + roi  # E_v1
        viz = visdom.Visdom(env=viz_name)  # 新建名为'E_v1'的环境
        args = get_args()


        if args.load:
            print('hello')
        else:
            if args.gpu:  # 默认为True
                net.cuda()

            try:
                train_net(net,  # 稀疏全连接网络
                          args.epochs,  # 50
                          args.batchsize,  # 64
                          args.lr,  # 1e-3
                          GaborNetDataset_trn_L,  # 训练数据
                          GaborNetDataset_val_L,  # 测试数据
                          # GaborNetDataset_test_L,
                          args.num,  # 300
                          num_v,  # 体素数量
                          viz,  # 一个可视化的啥玩意
                          roi,  # v1
                          sub_str,
                          gpu=args.gpu)  # True

            except KeyboardInterrupt:
                torch.save(net.state_dict(), 'INTERRUPTED.pth')
                print('Saved interrupt')
                try:
                    sys.exit(0)
                except SystemExit:
                    os._exit(0)
