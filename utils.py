#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import iid, noniid, sampling_num_data_iid
import pickle
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from networks.cls_models import *
import random 
import torchvision.models as premodels
from skimage import feature
from skimage.morphology import disk, dilation
import nibabel as nib
from medpy import metric
from scipy.spatial.distance import directed_hausdorff
from medpy.metric.binary import dc, hd
from scipy import ndimage
import custom_transform as trans
from glob import glob
import medpy.metric.binary as medmetric
from skimage.measure import find_contours
from skimage.measure import label, regionprops
import scipy


class Datum:
    def __init__(self, impath="", label=0, domain=0, classname=""):
        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname

class DomainDatasetBase(Dataset):
    def __init__(self, domainname, train=True, transform=None):
        self.root_dir = "../data/domainnet/"
        self.sub_directory = self.root_dir + domainname
        self.domain = domainname
        self.image_path = self.root_dir + f"{domainname}_train.txt" if train else self.root_dir + f"{domainname}_test.txt"
        self.data = self.readdata()
        self.transform = transform
        #self.images = torch.load(f'../data/domainnet/{domainname}_train.pt') if train else torch.load(f'../data/domainnet/{domainname}_test.pt')
        
    def readdata(self):
        items = []
        with open(self.image_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                impath, label = line.split(" ")
                classname = impath.split("/")[1]
                impath = os.path.join(self.root_dir, impath)
                label = int(label)
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=self.domain,
                    classname=classname
                )
                items.append(item)
        return items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx].impath).convert('RGB')
        label = self.data[idx].label

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class OfficehomeDatasetBase(Dataset):
    def __init__(self, domainname, train=True, transform=None):
        self.root_dir = "../data/officehome/"
        self.sub_directory = self.root_dir + domainname
        self.domain = domainname
        self.transform = transform
        self.images, self.labels = torch.load(f'../data/officehome/{domainname}_train.pt') if train else torch.load(f'../data/officehome/{domainname}_test.pt')
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class PACSDatasetBase(Dataset):
    def __init__(self, domainname, train=True, transform=None, transform2=None):
        self.root_dir = "../data/pacs/"
        self.sub_directory = self.root_dir + domainname
        self.domain = domainname
        self.transform = transform
        self.transform2 = transform2
        self.images, self.labels = torch.load(f'../data/pacs/{domainname}_train.pt') if train else torch.load(f'../data/pacs/{domainname}_test.pt')
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)
            if self.transform2 is not None:
                image2 = self.transform2(image)
                return image, image2, label
            return image, label
        

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        label = torch.as_tensor(np.array(label), dtype=torch.int64)
        return torch.as_tensor(image), torch.as_tensor(label)


class FundusDatasetBase(Dataset):
    def __init__(self, domainname, train=True, transform=None):
        self.root_dir = "../data/fundus/"
        self.sub_directory = self.root_dir + domainname
        self.domain = domainname
        self.transform = transform
        self.images, self.labels = torch.load(f'../data/fundus/{domainname}_train2.pt') if train else torch.load(f'../data/fundus/{domainname}_test2.pt')
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)
            
            return image, label


class FundusSegmentation(Dataset):
    """
    Fundus segmentation dataset
    including 5 domain dataset
    one for test others for training
    """

    def __init__(self,
                 base_dir,
                 dataset='refuge',
                 split='train',
                 testid=None,
                 transform=None
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        self._base_dir = base_dir
        self.image_list = []
        self.split = split

        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []
        self.domain = dataset
        self._image_dir = os.path.join(self._base_dir, dataset, split, 'image')
        imagelist = glob(self._image_dir + "/*.png")
        for image_path in imagelist:
            gt_path = image_path.replace('image', 'mask')
            self.image_list.append({'image': image_path, 'label': gt_path, 'id': testid})

        self.transform = transform
        # self._read_img_into_memory()
        # Display stats
        print('| Number of {} in {} = {:d}'.format(self.domain, split, len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        _img = Image.open(self.image_list[index]['image']).convert('RGB')
        _target = Image.open(self.image_list[index]['label'])
        if _target.mode == 'RGB':
            _target = _target.convert('L')
        _img_name = self.image_list[index]['image'].split('/')[-1]

        # _img = self.image_pool[index]
        # _target = self.label_pool[index]
        # _img_name = self.img_name_pool[index]

        anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name}

        if self.transform is not None:
            anco_sample = self.transform(anco_sample)
            # if 'train' in self.split:
            #     single_channel_label = torch.argmax(anco_sample['label'], dim=0)

            #     background_mask = (single_channel_label == 0) & (anco_sample['label'].sum(dim=0) == 0)
            #     single_channel_label[background_mask] = 255
            #     return anco_sample['image'], single_channel_label
            # else:
            #     return anco_sample['image'], anco_sample['label']
            single_channel_label = torch.argmax(anco_sample['label'], dim=0)
            background_mask = (single_channel_label == 0) & (anco_sample['label'].sum(dim=0) == 0)
            single_channel_label[background_mask] = 255
            return anco_sample['image'], anco_sample['label']

    def _read_img_into_memory(self):

        img_num = len(self.image_list)
        for index in range(img_num):
            self.image_pool.append(Image.open(self.image_list[index]['image']).convert('RGB'))
            _target = Image.open(self.image_list[index]['label'])
            if _target.mode == 'RGB':
                _target = _target.convert('L')
            self.label_pool.append(_target)
            _img_name = self.image_list[index]['image'].split('/')[-1]
            self.img_name_pool.append(_img_name)


    def __str__(self):
        return 'Fundus(split=' + str(self.split) + ')'

class FourierTransform(nn.Module):
    def __init__(self, mode='amp'):
        super().__init__()
        self.mode = mode

    def forward(self, img):
        f_image = torch.fft.fftn(img)
        f_image_shifted = torch.fft.fftshift(f_image)
        phase = torch.angle(f_image_shifted)
        amplitude = torch.abs(f_image_shifted)
        if self.mode == 'amp':
            f_image_shifted = amplitude * torch.exp((1j*torch.randn_like(phase)))
        else:
            f_image_shifted = torch.randn_like(amplitude) * torch.exp((1j*phase))
        recon = torch.fft.ifftshift(f_image_shifted)
        recon = torch.fft.ifftn(recon)
        return torch.real(recon)

class FundusSegmentation2(Dataset):
    """
    Fundus segmentation dataset
    including 5 domain dataset
    one for test others for training
    """

    def __init__(self,
                 base_dir,
                 phase='train',
                 splitid=[2, 3, 4],
                 transform=None,
                 state='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        self.state = state
        self._base_dir = base_dir
        self.image_list = []
        self.phase = phase
        self.image_pool = {'DGS':[], 'REF':[], 'RIM':[], 'REF_val':[]}
        self.label_pool = {'DGS':[], 'REF':[], 'RIM':[], 'REF_val':[]}
        self.img_name_pool = {'DGS':[], 'REF':[], 'RIM':[], 'REF_val':[]}

        self.flags_DGS = ['gd', 'nd']
        self.flags_REF = ['g', 'n']
        self.flags_RIM = ['G', 'N', 'S']
        self.flags_REF_val = ['V']
        self.splitid = splitid
        SEED = 1212
        random.seed(SEED)
        for id in splitid:
            self._image_dir = os.path.join(self._base_dir, 'Domain'+str(id), phase, 'ROIs/image/')
            print('==> Loading {} data from: {}'.format(phase, self._image_dir))

            imagelist = glob(self._image_dir + '*.png')
            for image_path in imagelist:
                gt_path = image_path.replace('image', 'mask')
                self.image_list.append({'image': image_path, 'label': gt_path})

        self.transform = transform
        self._read_img_into_memory()
        for key in self.image_pool:
            if len(self.image_pool[key]) < 1:
                del self.image_pool[key]
                del self.label_pool[key]
                del self.img_name_pool[key]
                break
        for key in self.image_pool:
            if len(self.image_pool[key]) < 1:
                del self.image_pool[key]
                del self.label_pool[key]
                del self.img_name_pool[key]
                break
        for key in self.image_pool:
            if len(self.image_pool[key]) < 1:
                del self.image_pool[key]
                del self.label_pool[key]
                del self.img_name_pool[key]
                break
        # Display stats
        print('-----Total number of images in {}: {:d}'.format(phase, len(self.image_list)))

    def __len__(self):
        max = -1
        for key in self.image_pool:
             if len(self.image_pool[key])>max:
                 max = len(self.image_pool[key])
        return max

    def __getitem__(self, index):
        if self.phase != 'test':
            sample = []
            for key in self.image_pool:
                domain_code = list(self.image_pool.keys()).index(key)
                index = np.random.choice(len(self.image_pool[key]), 1)[0]
                _img = self.image_pool[key][index]
                _target = self.label_pool[key][index]
                _img_name = self.img_name_pool[key][index]
                anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name, 'dc': domain_code}
                if self.transform is not None:
                    anco_sample = self.transform(anco_sample)
                sample.append(anco_sample)
        else:
            sample = []
            for key in self.image_pool:
                domain_code = list(self.image_pool.keys()).index(key)
                _img = self.image_pool[key][index]
                _target = self.label_pool[key][index]
                _img_name = self.img_name_pool[key][index]
                anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name, 'dc': domain_code}
                if self.transform is not None:
                    anco_sample = self.transform(anco_sample)
                sample=anco_sample
        return sample

    def _read_img_into_memory(self):
        img_num = len(self.image_list)
        for index in range(img_num):
            basename = os.path.basename(self.image_list[index]['image'])
            Flag = "NULL"
            if basename[0:2] in self.flags_DGS:
                Flag = 'DGS'
            elif basename[0] in self.flags_REF:
                Flag = 'REF'
            elif basename[0] in self.flags_RIM:
                Flag = 'RIM'
            elif basename[0] in self.flags_REF_val:
                Flag = 'REF_val'
            else:
                print("[ERROR:] Unknown dataset!")
                return 0
            if self.splitid[0] == '4':
                # self.image_pool[Flag].append(Image.open(self.image_list[index]['image']).convert('RGB').resize((256, 256), Image.LANCZOS))
                self.image_pool[Flag].append(Image.open(self.image_list[index]['image']).convert('RGB').crop((144, 144, 144+512, 144+512)).resize((256, 256), Image.LANCZOS))
                _target = np.asarray(Image.open(self.image_list[index]['label']).convert('L'))
                _target = _target[144:144+512, 144:144+512]
                _target = Image.fromarray(_target)
            else:
                self.image_pool[Flag].append(
                    Image.open(self.image_list[index]['image']).convert('RGB').resize((256, 256), Image.LANCZOS))
                # self.image_pool[Flag].append(Image.open(self.image_list[index]['image']).convert('RGB'))
                _target = Image.open(self.image_list[index]['label'])

            if _target.mode is 'RGB':
                _target = _target.convert('L')
            if self.state != 'prediction':
                _target = _target.resize((256, 256))
            # print(_target.size)
            # print(_target.mode)
            self.label_pool[Flag].append(_target)
            # if self.split[0:4] in 'test':
            _img_name = self.image_list[index]['image'].split('/')[-1]
            self.img_name_pool[Flag].append(_img_name)


def get_dataset(args):
    if args.dataset == 'pacs':
        transform_train = transforms.Compose([
            transforms.Resize((224,224), antialias=True),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        

        transform_test = transforms.Compose([
            #transforms.ToTensor(),
            transforms.Resize((224,224),antialias=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_loaders = []
        test_loaders = []
        domains = ['photo', 'art_painting', 'sketch', 'cartoon']
        count = 0
        for domain in domains:
            test_dataset = PACSDatasetBase(domain, False, transform_test)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
            test_loaders.append(test_loader)
            if args.target_domain == domain:
                continue
            if args.alg == 'fedde':
                transform_fourier = FourierTransform(mode='amp')
                dataset = PACSDatasetBase(domain, True, transform_train)
            else:
                dataset = PACSDatasetBase(domain, True, transform_train)
            train_loaders.append(DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4))
    elif args.dataset == 'domainnet':
        transform_train = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                std=[x/255.0 for x in [63.0, 62.1, 66.7]]),
        ])
        
        train_loaders = []
        test_loaders = []
        domains = ['sketch', 'clipart', 'infograph', 'real', 'painting', 'quickdraw']
        for domain in domains:
            test_dataset = DomainDatasetBase(domain, False, transform_test)
            test_loaders.append(DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4))
            if domain == args.target_domain:
                continue
            dataset = DomainDatasetBase(domain, True, transform_train)
            train_loaders.append(DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4))
    elif args.dataset == 'officehome':
        transform_train = transforms.Compose([
            transforms.Resize((224,224), antialias=True),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        train_loaders = []
        test_loaders = []
        domains = ['art', 'clipart', 'product', 'realworld']
        for domain in domains:
            test_dataset = OfficehomeDatasetBase(domain, False, transform_test)
            test_loaders.append(DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4))
            if domain == args.target_domain:
                continue
            dataset = OfficehomeDatasetBase(domain, True, transform_train)
            train_loaders.append(DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4))
    elif args.dataset == 'prostate':
        pass
    elif args.dataset == 'fundus':
        transform_train = transforms.Compose([
            trans.RandomScaleCrop(256),
            #trans.RandomRotate(),
            #trans.RandomFlip(),
            #trans.elastic_transform(),
            #trans.add_salt_pepper_noise(),
            #trans.adjust_light(),
            #trans.eraser(),
            trans.Normalize_tf(),
            trans.ToTensor()
        ])

        transform_test = transforms.Compose([
            trans.Normalize_tf(),  # this function separates (raw ground truth mask) into (2 masks)
            trans.ToTensor()
        ])
        
        train_loaders = []
        test_loaders = []
        domains = ['1', '2', '3', '4']
        for domain in domains:
            #test_dataset = FundusSegmentation(base_dir='../data/fundus/', dataset=domain, split='test/ROIs', transform=transform_test)
            test_dataset = FundusSegmentation2(base_dir='../../data/fundus/', phase='test', splitid=[domain],
                                    transform=transform_test, state='prediction')
            #test_dataset = FundusDatasetBase(domain, False, transform_test)
            test_loaders.append(DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True))
            if domain == args.target_domain:
                continue
            #dataset = FundusSegmentation(base_dir='../data/fundus/', dataset=domain, split='train/ROIs', transform=transform_train)
            dataset = FundusSegmentation2(base_dir='../../data/fundus/', phase='train', splitid=[domain],
                                                         transform=transform_train)
            #dataset = FundusDatasetBase(domain, True, transform_train)
            train_loaders.append(DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True))
            
    return train_loaders, test_loaders, domains

def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg    

def postprocessing(prediction, threshold=0.75, dataset='G'):
    if dataset[0] == 'D':
        # prediction = prediction.numpy()
        prediction_copy = np.copy(prediction)
        disc_mask = prediction[1]
        cup_mask = prediction[0]
        disc_mask = (disc_mask > 0.5)  # return binary mask
        cup_mask = (cup_mask > 0.1)  # return binary mask
        disc_mask = disc_mask.astype(np.uint8)
        cup_mask = cup_mask.astype(np.uint8)
        # for i in range(5):
        #     disc_mask = scipy.signal.medfilt2d(disc_mask, 7)
        #     cup_mask = scipy.signal.medfilt2d(cup_mask, 7)
        # disc_mask = morphology.binary_erosion(disc_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
        # cup_mask = morphology.binary_erosion(cup_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
        disc_mask = get_largest_fillhole(disc_mask).astype(np.uint8)  # return 0,1
        cup_mask = get_largest_fillhole(cup_mask).astype(np.uint8)
        prediction_copy[0] = cup_mask
        prediction_copy[1] = disc_mask
        return prediction_copy
    else:
        prediction = torch.sigmoid(prediction).data.cpu().numpy()

        # disc_mask = scipy.signal.medfilt2d(disc_mask, 7)
        # cup_mask = scipy.signal.medfilt2d(cup_mask, 7)
        # disc_mask = morphology.erosion(disc_mask, morphology.diamond(3))  # return 0,1
        # cup_mask = morphology.erosion(cup_mask, morphology.diamond(3))  # return 0,1

        prediction_copy = np.copy(prediction)
        prediction_copy = (prediction_copy > threshold)  # return binary mask
        prediction_copy = prediction_copy.astype(np.uint8)
        disc_mask = prediction_copy[1]
        cup_mask = prediction_copy[0]
        disc_mask = get_largest_fillhole(disc_mask).astype(np.uint8)  # return 0,1
        cup_mask = get_largest_fillhole(cup_mask).astype(np.uint8)
        prediction_copy[0] = cup_mask
        prediction_copy[1] = disc_mask
        # selem = disk(6)
        # disc_mask = morphology.closing(disc_mask, selem)
        # cup_mask = morphology.closing(cup_mask, selem)
        # print(sum(disc_mask))


        return prediction_copy

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass
    
def dice_loss1(score, target):
    target = target.float()
    smooth = 1.
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def dice_loss(score, target):
    smooth = 1e-5
    iflat = score.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))

def entropy_loss(p,C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice

def cross_entropy2d(logit, target, ignore_index=255, weight=None, size_average=True, batch_average=True):
    n, c, h, w = logit.size()
    # logit = logit.permute(0, 2, 3, 1)
    target = target.squeeze(1)
    if weight is None:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, size_average=False)
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().cuda(), ignore_index=ignore_index, size_average=False)
    loss = criterion(logit, target.long())

    if size_average:
        loss /= (h * w)

    if batch_average:
        loss /= n

    return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

def connectivity_region_analysis(mask):
    s = [[0,1,0],
         [1,1,1],
         [0,1,0]]
    label_im, nb_labels = ndimage.label(mask)#, structure=s)

    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))

    # plt.imshow(label_im)        
    label_im[label_im != np.argmax(sizes)] = 0
    label_im[label_im == np.argmax(sizes)] = 1

    return label_im

def replace_weight(w, r_w):
    for key2 in r_w.keys():
        w[key2] = r_w[key2]
    return w

def check_param(w, r_w):
    for k1, k2 in zip(w.keys(), r_w.keys()):
        print(k1, k2)
        w[k1] = r_w[k2]
    return w

def dice_loss3(pred, target, smooth=1e-6):
    # 예측을 softmax로 두 클래스 확률로 변환
    pred = F.softmax(pred, dim=1)  # (bs, 2, 384, 384)
    
    # 각 클래스에 대해 Dice Loss 계산
    dice_loss_total = 0
    for cls in range(2):  # 클래스 0과 클래스 1에 대해 반복
        pred_cls = pred[:, cls, :, :]  # 해당 클래스의 예측 (bs, 384, 384)
        target_cls = (target == cls).float()  # 해당 클래스의 실제 값 (bs, 384, 384)

        # Dice Coefficient 계산
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice_loss_cls = 1 - (2 * intersection + smooth) / (union + smooth)
        
        dice_loss_total += dice_loss_cls  # 두 클래스의 Dice Loss 합산
    
    # 평균 Dice Loss 계산
    return dice_loss_total / 2

def get_largest_fillhole(binary):
    label_image = label(binary)
    regions = regionprops(label_image)
    area_list = []
    for region in regions:
        area_list.append(region.area)
    if area_list:
        idx_max = np.argmax(area_list)
        binary[label_image != idx_max + 1] = 0
    return scipy.ndimage.binary_fill_holes(np.asarray(binary).astype(int))

def concat_weights(w):
    w_concat = copy.deepcopy(w[0])
    for key in w_concat.keys():
        for i in range(1, len(w)):
            w_concat[key] = torch.cat((w_concat[key], w[i][key]), dim=0)
        
    return w_concat

def bias_reshape(w, size):
    for key in w.keys():
        
        if 'bn' in key or 'downsample.1.bias' in key:
            w[key] = torch.div(torch.sum(w[key].view(size, -1), dim=0), size)
        elif 'bias' in key:
            w[key] = w[key].view(size, -1)
    return w

def change_weight(w1, w2):
    weight_ = []
    for k in w2.keys():
        if 'alpha' not in k and 'gamma' not in k and 'linear' not in k:
            weight_.append(w2[k])
    for idx, k in enumerate(w1.keys()):
        if idx == len(weight_):
            break
        w1[k] = weight_[idx]
    return w1

def getNetwork(args):
    if args.dataset == 'domainnet':
        fcdim = 345
    elif args.dataset == 'pacs':
        fcdim = 7
    elif args.dataset == 'vlcs':
        fcdim = 5
    elif args.dataset == 'officehome':
        fcdim = 65
    elif args.dataset == 'fundus':
        fcdim = 2
    else:
        fcdim = 10
    models = []
    if args.alg == 'fedavg':
        net = BaseModel(args.model, args.out_dim, fcdim)
        global_model = BaseModel(args.model, args.out_dim, fcdim)
    elif args.alg == 'fedde':
        net = Ensemble_Model(args.model, args.out_dim, fcdim, num_models=1)
        global_model = Ensemble_Model(args.model, args.out_dim, fcdim, num_models=args.num_users)
        for i in range(args.num_users):
            models.append(copy.deepcopy(net))
    elif args.alg == 'feddc':
        net = Ensemble_Model(args.model, args.out_dim, fcdim, num_models=1)
        #global_model = NonGlobalModel(args.model, fcdim)
        global_model = Ensemble_Model(args.model, args.out_dim, fcdim, num_models=args.num_users)
        for i in range(args.num_users):
            models.append(copy.deepcopy(net))
    elif args.alg == 'fedsr':
        net = SRModel(args.model, args.out_dim, fcdim)
        global_model = SRModel(args.model, args.out_dim, fcdim)
    non_global_model = NonGlobalModel(args.model, fcdim)
    return models, global_model, non_global_model
