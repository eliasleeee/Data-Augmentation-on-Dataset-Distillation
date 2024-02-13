import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision
from scipy.ndimage.interpolation import rotate as scipyrotate
from networks import MLP, ConvNet, LeNet, AlexNet, AlexNetBN, VGG11, VGG11BN, ResNet18, ResNet18BN_AP, ResNet18BN

def get_dataset(dataset, data_path):
    if dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'FashionMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.2861]
        std = [0.3530]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'SVHN':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.SVHN(data_path, split='train', download=True, transform=transform)  # no augmentation
        dst_test = datasets.SVHN(data_path, split='test', download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'CIFAR100':
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.5071, 0.4866, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'TinyImageNet':
        channel = 3
        im_size = (64, 64)
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        data = torch.load(os.path.join(data_path, 'tinyimagenet.pt'), map_location='cpu')

        class_names = data['classes']

        images_train = data['images_train']
        labels_train = data['labels_train']
        images_train = images_train.detach().float() / 255.0
        labels_train = labels_train.detach()
        for c in range(channel):
            images_train[:,c] = (images_train[:,c] - mean[c])/std[c]
        dst_train = TensorDataset(images_train, labels_train)  # no augmentation

        images_val = data['images_val']
        labels_val = data['labels_val']
        images_val = images_val.detach().float() / 255.0
        labels_val = labels_val.detach()

        for c in range(channel):
            images_val[:, c] = (images_val[:, c] - mean[c]) / std[c]

        dst_test = TensorDataset(images_val, labels_val)  # no augmentation

    else:
        exit('unknown dataset: %s'%dataset)


    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader



class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]
    

class Cutout(object):
    def __init__(self, length, prob=1.0):
        self.length = length
        self.prob = prob

    def __call__(self, img):
        if np.random.binomial(1, self.prob):
            h, w = img.size(2), img.size(3)
            mask = np.ones((h, w), np.float32)
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.
            mask = torch.from_numpy(mask).to('cuda')
            mask = mask.expand_as(img)
            img *= mask
        return img


    
def custom_aug(images, args):
    image_syn_vis = images.clone()
    if args.normalize_data:
        if args.dataset == 'CIFAR10':
            channel = 3
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        elif args.dataset == 'CIFAR100':
            channel = 3
            mean = [0.5071, 0.4866, 0.4409]
            std = [0.2673, 0.2564, 0.2762]
        elif args.dataset == 'tinyimagenet':
            channel = 3
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif args.dataset == 'SVHN':    
            channel = 3
            mean = [0.4377, 0.4438, 0.4728]
            std = [0.1980, 0.2010, 0.1970]
        elif args.dataset == 'MNIST':    
            channel = 1
            mean = [0.1307]
            std = [0.3081]


        for ch in range(channel):
            image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
    image_syn_vis[image_syn_vis<0] = 0.0
    image_syn_vis[image_syn_vis>1] = 1.0

    normalized_d = image_syn_vis * 255
    if args.dataset == 'tinyimagenet':
        size = 64
    else:
        size = 32
    if args.aug == 'autoaug':
        if args.dataset == 'tinyimagenet':
            data_transforms = transforms.Compose([transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.IMAGENET)])
        elif args.dataset == 'CIFAR10':
            data_transforms = transforms.Compose([transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10)])
        elif args.dataset == 'SVHN':
            data_transforms = transforms.Compose([transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.SVHN)])

    elif args.aug == 'randaug':
        data_transforms = transforms.Compose([transforms.RandAugment(num_ops=1)])
    elif args.aug == 'imagenetaug':
        data_transforms = transforms.Compose([transforms.RandomCrop(size, padding=4), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)])
    elif args.aug == 'cifaraug':
        data_transforms = transforms.Compose([transforms.RandomCrop(size, padding=4), transforms.RandomHorizontalFlip()])
    else:
        exit('unknown augmentation method: %s'%args.aug)
    normalized_d = data_transforms(normalized_d.to(torch.uint8))
    normalized_d = normalized_d / 255.0

    # print("changes after autoaug: ", (normalized_d - image_syn_vis).pow(2).sum().item())

    if args.normalize_data:
        for ch in range(channel):
            normalized_d[:, ch] = (normalized_d[:, ch] - mean[ch])  / std[ch]

    if args.aug == 'cifar_aug':
        cutout_transform = transforms.Compose([Cutout(16, 1)])
        normalized_d = cutout_transform(normalized_d)

    return normalized_d



def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling



def get_network(model, seed, channel, num_classes, im_size=(32, 32)):
    torch.random.manual_seed(seed)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes)
    elif model == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNet':
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNetBN':
        net = AlexNetBN(channel=channel, num_classes=num_classes)
    elif model == 'VGG11':
        net = VGG11( channel=channel, num_classes=num_classes)
    elif model == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN_AP':
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN':
        net = ResNet18BN(channel=channel, num_classes=num_classes)

    elif model == 'ConvNetD1':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=1, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD2':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=2, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD3':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=3, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD4':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=4, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetW32':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=32, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetW64':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=64, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetW128':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetW256':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=256, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetAS':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='sigmoid', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetAR':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='relu', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetAL':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='leakyrelu', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetASwish':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='swish', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetASwishBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='swish', net_norm='batchnorm', net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetNN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='none', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='batchnorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetLN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='layernorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetIN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='instancenorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetGN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='groupnorm', net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetNP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='none', im_size=im_size)
    elif model == 'ConvNetMP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='maxpooling', im_size=im_size)
    elif model == 'ConvNetAP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='avgpooling', im_size=im_size)

    else:
        net = None
        exit('unknown model: %s'%model)

    gpu_num = torch.cuda.device_count()
    if gpu_num>0:
        device = 'cuda'
        if gpu_num>1:
            net = nn.DataParallel(net)
    else:
        device = 'cpu'
    net = net.to(device)

    return net



def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))



def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis



def match_loss(gw_syn, gw_real, args):
    dis = torch.tensor(0.0).to(args.device)

    if args.dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('unknown distance function: %s'%args.dis_metric)

    return dis



def get_loops(ipc):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if ipc == 1:
        outer_loop, inner_loop = 1, 1
    elif ipc == 10:
        outer_loop, inner_loop = 10, 50
    elif ipc == 20:
        outer_loop, inner_loop = 20, 25
    elif ipc == 30:
        outer_loop, inner_loop = 30, 20
    elif ipc == 40:
        outer_loop, inner_loop = 40, 15
    elif ipc == 50:
        outer_loop, inner_loop = 50, 10
    elif ipc == 5000:
        outer_loop, inner_loop = 50, 10
    else:
        outer_loop, inner_loop = 0, 0
        exit('loop hyper-parameters are not defined for %d ipc'%ipc)
    return outer_loop, inner_loop



def epoch(mode, dataloader, net, optimizer, criterion, args, aug, stage):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)
    criterion = criterion.to(args.device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    if stage == 1:
        args.dsa_param = args.dsa_param_stage_1
        args.dsa_strategy = args.dsa_strategy_stage_1
    elif stage == 2:
        args.dsa_param = args.dsa_param_stage_2
        args.dsa_strategy = args.dsa_strategy_stage_2


    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(args.device)
        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            else:
                img = augment(img, args.dc_aug_param, device=args.device)
        lab = datum[1].long().to(args.device)
        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg



def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, stage):
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    labels_train = labels_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(args.device)

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    if stage == 0:
        args.dsa_param = args.dsa_param_stage_0
        args.dsa_strategy = args.dsa_strategy_stage_0
    elif stage == 1:
        args.dsa_param = args.dsa_param_stage_1
        args.dsa_strategy = args.dsa_strategy_stage_1
    elif stage == 2:
        args.dsa_param = args.dsa_param_stage_2
        args.dsa_strategy = args.dsa_strategy_stage_2
    
    start = time.time()
    for ep in range(Epoch+1):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, aug = True, stage=stage )
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    time_train = time.time() - start
    loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug = False, stage=stage)
    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

    return net, acc_train, acc_test



def augment(images, dc_aug_param, device):
    # This can be sped up in the future.

    if dc_aug_param != None and dc_aug_param['strategy'] != 'none':
        scale = dc_aug_param['scale']
        crop = dc_aug_param['crop']
        rotate = dc_aug_param['rotate']
        noise = dc_aug_param['noise']
        strategy = dc_aug_param['strategy']

        shape = images.shape
        mean = []
        for c in range(shape[1]):
            mean.append(float(torch.mean(images[:,c])))

        def cropfun(i):
            im_ = torch.zeros(shape[1],shape[2]+crop*2,shape[3]+crop*2, dtype=torch.float, device=device)
            for c in range(shape[1]):
                im_[c] = mean[c]
            im_[:, crop:crop+shape[2], crop:crop+shape[3]] = images[i]
            r, c = np.random.permutation(crop*2)[0], np.random.permutation(crop*2)[0]
            images[i] = im_[:, r:r+shape[2], c:c+shape[3]]

        def scalefun(i):
            h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            tmp = F.interpolate(images[i:i + 1], [h, w], )[0]
            mhw = max(h, w, shape[2], shape[3])
            im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
            r = int((mhw - h) / 2)
            c = int((mhw - w) / 2)
            im_[:, r:r + h, c:c + w] = tmp
            r = int((mhw - shape[2]) / 2)
            c = int((mhw - shape[3]) / 2)
            images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

        def rotatefun(i):
            im_ = scipyrotate(images[i].cpu().data.numpy(), angle=np.random.randint(-rotate, rotate), axes=(-2, -1), cval=np.mean(mean))
            r = int((im_.shape[-2] - shape[-2]) / 2)
            c = int((im_.shape[-1] - shape[-1]) / 2)
            images[i] = torch.tensor(im_[:, r:r + shape[-2], c:c + shape[-1]], dtype=torch.float, device=device)

        def noisefun(i):
            images[i] = images[i] + noise * torch.randn(shape[1:], dtype=torch.float, device=device)


        augs = strategy.split('_')

        for i in range(shape[0]):
            choice = np.random.permutation(augs)[0] # randomly implement one augmentation
            if choice == 'crop':
                cropfun(i)
            elif choice == 'scale':
                scalefun(i)
            elif choice == 'rotate':
                rotatefun(i)
            elif choice == 'noise':
                noisefun(i)

    return images



def get_daparam(dataset, model, model_eval, ipc):
    # We find that augmentation doesn't always benefit the performance.
    # So we do augmentation for some of the settings.

    dc_aug_param = dict()
    dc_aug_param['crop'] = 4
    dc_aug_param['scale'] = 0.2
    dc_aug_param['rotate'] = 45
    dc_aug_param['noise'] = 0.001
    dc_aug_param['strategy'] = 'none'

    if dataset == 'MNIST':
        dc_aug_param['strategy'] = 'crop_scale_rotate'

    if model_eval in ['ConvNetBN']: # Data augmentation makes model training with Batch Norm layer easier.
        dc_aug_param['strategy'] = 'crop_noise'

    return dc_aug_param


def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == 'M': # multiple architectures
        model_eval_pool = ['MLP', 'ConvNet', 'LeNet', 'AlexNet', 'VGG11', 'ResNet18']
    elif eval_mode == 'B':  # multiple architectures with BatchNorm for DM experiments
        model_eval_pool = ['ConvNetBN', 'ConvNetASwishBN', 'AlexNetBN', 'VGG11BN', 'ResNet18BN']
    elif eval_mode == 'W': # ablation study on network width
        model_eval_pool = ['ConvNetW32', 'ConvNetW64', 'ConvNetW128', 'ConvNetW256']
    elif eval_mode == 'D': # ablation study on network depth
        model_eval_pool = ['ConvNetD1', 'ConvNetD2', 'ConvNetD3', 'ConvNetD4']
    elif eval_mode == 'A': # ablation study on network activation function
        model_eval_pool = ['ConvNetAS', 'ConvNetAR', 'ConvNetAL', 'ConvNetASwish']
    elif eval_mode == 'P': # ablation study on network pooling layer
        model_eval_pool = ['ConvNetNP', 'ConvNetMP', 'ConvNetAP']
    elif eval_mode == 'N': # ablation study on network normalization layer
        model_eval_pool = ['ConvNetNN', 'ConvNetBN', 'ConvNetLN', 'ConvNetIN', 'ConvNetGN']
    elif eval_mode == 'S': # itself
        if 'BN' in model:
            print('Attention: Here I will replace BN with IN in evaluation, as the synthetic set is too small to measure BN hyper-parameters.')
        model_eval_pool = [model[:model.index('BN')]] if 'BN' in model else [model]
    elif eval_mode == 'SS':  # itself
        model_eval_pool = [model]
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool


class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.brightness = 1.0
        self.new = 1.0 # scale from 0 to 1.0
        self.new1 = 1.0 # scale from 0 to 1.0
        self.new2 = 4 
        self.nia1 = 1.0 
        self.saturation = 2.0
        self.contrast = 0.5
        self.ratio_shearing = 0.5 # the size would be 0.5x0.5


def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1


def DiffAugment(x, strategy='', seed = -1, param = None):
    if strategy == 'None' or strategy == 'none' or strategy == '':
        return x

    if seed == -1:
        param.Siamese = False
    else:
        param.Siamese = True

    param.latestseed = seed
    

    if strategy:
        if param.aug_mode == 'M': # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('unknown augmentation mode: %s'%param.aug_mode)
        x = x.contiguous()
    return x


# We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0].clone()
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


# def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
#     ratio = param.ratio_rotate
#     set_seed_DiffAug(param)
#     theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
#     theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
#         [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
#     theta = torch.tensor(theta, dtype=torch.float)
#     if param.Siamese: # Siamese augmentation:
#         theta[:] = theta[0].clone()
#     grid = F.affine_grid(theta, x.shape).to(x.device)
#     x = F.grid_sample(x, grid)
#     return x

def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0].clone()
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x

def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.Siamese: # Siamese augmentation:
        randf[:] = randf[0].clone()
    return torch.where(randf < prob, x.flip(3), x)



def rand_new(x, param):
    ratio = param.new
    set_seed_DiffAug(param)
    mask = torch.zeros(x.size(0), x.size(1), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    
    for i in range(x.size(2)):
        mask[:,:,i,:] = (i - x.size(2)/2)/ x.size(2)

    if param.Siamese:  # Siamese augmentation:
        mask[:] = mask[0].clone()
    x = x + mask*ratio*2
    return x 

def rand_new1(x, param):
    ratio = param.new1
    set_seed_DiffAug(param)
    c0_mean = x.mean(dim=1, keepdim=True)
    c1_mean = x.mean(dim=1, keepdim=True)
    c2_mean = x.mean(dim=1, keepdim=True)
    
    randn = torch.randint(x.size(0), x.size(1), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    
    mask[:,2,:,:] = 0.5

    if param.Siamese:  # Siamese augmentation:
        randn[:] = randn[0].clone()
    
    x = x * mask

    return x 

def rand_rotatedegree(x, param):
    ratio = param.rotate_degree
    theta = torch.ones(x.shape[0])  * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0].clone()
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)

    return x

def get_mask(mask,maskf,length,ratio,randarea,offset_x,offset_y):
    
    cutout_size = int(lengtt*ratio*0.5)
    
    if randarea == 0: 
        for i in range(cutout_size):
            mask[:,0+(offset_x):cutout_size-i-1+(offset_x),i+(offset_y)] = 0
            maskf[:,0+(offset_x):cutout_size-i-1+(offset_x),i+(offset_y)] = 1
    elif randarea == 1:
        for i in range(cutout_size):
            mask[:,lengtt-cutout_size+i-(offset_x):lengtt-(offset_x),i+(offset_y)] = 0
            maskf[:,lengtt-cutout_size+i-(offset_x):lengtt-(offset_x),i+(offset_y)] = 1
    elif randarea == 2:
        for i in range(cutout_size):
            mask[:,lengtt-cutout_size+i-(offset_x):lengtt-(offset_x),lengtt-i-1-(offset_y)] = 0
            maskf[:,lengtt-cutout_size+i-(offset_x):lengtt-(offset_x),lengtt-i-1-(offset_y)] = 1
    elif randarea == 3:
        for i in range(cutout_size):
            mask[:,0+(offset_x):cutout_size-i-1+(offset_x),lengtt-i-1-(offset_y)] = 0 
            maskf[:,0+(offset_x):cutout_size-i-1+(offset_x),lengtt-i-1-(offset_y)] = 1     

    return mask, maskf


def get_mask_circle(mask,maskf,length,randradius_inside,randradius_outside,offset_x,offset_y):
    
    x_min = max(offset_x - randradius_outside,0)
    x_max = min(offset_x + randradius_outside,length)
    y_min = max(offset_y - randradius_outside,0) 
    y_max = min(offset_y + randradius_outside,length)

    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            distance = (i-offset_x)**2 + (j-offset_y)**2
            if distance < randradius_outside**2 and distance > randradius_inside**2:
                mask[:,i,j] = 0
                maskf[:,i,j] = 1
     
    return mask, maskf

def rand_nia1(x, param):
    ratio = 1.2
    lengtt = x.size(2)
    cutout_size = int(lengtt*ratio*0.5)

    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    maskf = torch.zeros(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    color = torch.zeros(x.size(0),x.size(1), 1, 1, dtype=x.dtype, device=x.device)
    set_seed_DiffAug(param)
    

    offset_x = torch.randint(0, x.size(2) - cutout_size -1, size=[1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) - cutout_size -1, size=[1], device=x.device)

    set_seed_DiffAug(param)
    #randarea1 = torch.randint(0,4,size=[1], device=x.device)
    randarea1 = torch.randint(0,4,size=[1])
    #set_seed_DiffAug(param)
    #randarea2 = torch.randint(0,4,size=[1], device=x.device)
    #set_seed_DiffAug(param)
    #randarea3 = torch.randint(0,4,size=[1], device=x.device)

    maskr1,maskf1 = get_mask(mask,maskf,lengtt,ratio,randarea1,offset_x,offset_y)
    #maskr2 = get_mask(mask,lengtt,ratio,randarea2)
    #maskr3 = get_mask(mask,lengtt,ratio,randarea3)

    color= torch.sum(x*maskf1.unsqueeze(1), dim=(2,3), keepdim = True) / torch.sum(maskf1.unsqueeze(1), dim=(2,3), keepdim = True)

    return x* maskr1.unsqueeze(1) + color*maskf1.unsqueeze(1)

def rand_nia2(x, param):
    
    length = x.size(2)

    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    maskf = torch.zeros(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    #color = torch.zeros(x.size(0),x.size(1), 1, 1, dtype=x.dtype, device=x.device)
    #set_seed_DiffAug(param)
    color = torch.rand(x.size(0), x.size(1), 1, 1, dtype=x.dtype, device=x.device)
    fullcolor = torch.ones(x.size(0), x.size(1), 1, 1, dtype=x.dtype, device=x.device)
    #noise = torch.rand(x.size(0), x.size(1), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    
    
    offset_range = param.offset_range_nia2
    
    while True:
        set_seed_DiffAug(param)
        offset_x = np.random.randint(x.size(2)/2 - 0.5 - offset_range,high = x.size(2)/2 -0.5 + offset_range)
        set_seed_DiffAug(param)
        offset_y = np.random.randint(x.size(3)/2 - 0.5 - offset_range,high = x.size(3)/2 -0.5 + offset_range)

        if ((offset_x-x.size(2)/2+0.5)**2 + (offset_y-x.size(3)/2+0.5)**2) <= (offset_range)**2 and offset_x%1==0 and offset_y%1==0:
            break
    


    radius_range = param.radius_range
    radius_min = param.radius_min
    radius_max = radius_min + radius_range

    thickness_range = param.thickness_range
    thickness_min = param.thickness_min
    thickness_max = thickness_min + thickness_range

    set_seed_DiffAug(param)
    randradius_inside = np.random.randint(radius_min, high = radius_max)
    set_seed_DiffAug(param)
    randradius_outside = np.random.randint(randradius_inside + thickness_min, high = randradius_inside + thickness_max )
   

    mask1,maskf1 = get_mask_circle(mask,maskf,length,randradius_inside,randradius_outside,offset_x,offset_y)
  
    #color= torch.sum(x*maskf1.unsqueeze(1), dim=(2,3), keepdim = True) / torch.sum(maskf1.unsqueeze(1), dim=(2,3), keepdim = True)

    return x*mask1.unsqueeze(1)  + color*maskf1.unsqueeze(1)
    #return x + fullcolor*maskf1.unsqueeze(1)


def rand_nia2a(x, param):
    
    length = x.size(2)

    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    maskf = torch.zeros(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    #color = torch.zeros(x.size(0),x.size(1), 1, 1, dtype=x.dtype, device=x.device)
    #set_seed_DiffAug(param)
    color = torch.rand(x.size(0), x.size(1), 1, 1, dtype=x.dtype, device=x.device)
    fullcolor = torch.ones(x.size(0), x.size(1), 1, 1, dtype=x.dtype, device=x.device)
    #noise = torch.rand(x.size(0), x.size(1), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    
    
    offset_range = param.offset_range_nia2
    
    while True:
        set_seed_DiffAug(param)
        offset_x = np.random.randint(x.size(2)/2 - 0.5 - offset_range,high = x.size(2)/2 -0.5 + offset_range)
        set_seed_DiffAug(param)
        offset_y = np.random.randint(x.size(3)/2 - 0.5 - offset_range,high = x.size(3)/2 -0.5 + offset_range)

        if ((offset_x-x.size(2)/2+0.5)**2 + (offset_y-x.size(3)/2+0.5)**2) <= (offset_range)**2 and offset_x%1==0 and offset_y%1==0:
            break
    


    radius_range = param.radius_range
    radius_min = param.radius_min
    radius_max = radius_min + radius_range

    thickness_range = param.thickness_range
    thickness_min = param.thickness_min
    thickness_max = thickness_min + thickness_range

    set_seed_DiffAug(param)
    randradius_inside = np.random.randint(radius_min, high = radius_max)
    set_seed_DiffAug(param)
    randradius_outside = np.random.randint(randradius_inside + thickness_min, high = randradius_inside + thickness_max )
   

    mask1,maskf1 = get_mask_circle(mask,maskf,length,randradius_inside,randradius_outside,offset_x,offset_y)
  
    #color= torch.sum(x*maskf1.unsqueeze(1), dim=(2,3), keepdim = True) / torch.sum(maskf1.unsqueeze(1), dim=(2,3), keepdim = True)

    return x*mask1.unsqueeze(1)  + color*maskf1.unsqueeze(1)
    #return x + fullcolor*maskf1.unsqueeze(1)




def rand_nia6(x, param):
    
    granularity1 = 1 
    granularity2 = 1 
    length = x.size(2)

    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    maskf = torch.zeros(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    #color = torch.zeros(x.size(0),x.size(1), 1, 1, dtype=x.dtype, device=x.device)
    #set_seed_DiffAug(param)
    color = torch.rand(x.size(0), x.size(1), 1, 1, dtype=x.dtype, device=x.device)
    fullcolor = torch.ones(x.size(0), x.size(1), 1, 1, dtype=x.dtype, device=x.device)
    #noise = torch.rand(x.size(0), x.size(1), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    
    
    offset_range = param.offset_range_nia2
    
    while True:

        set_seed_DiffAug(param)
        offset_x = np.random.randint(x.size(2)/2 - 0.5 - offset_range,high = x.size(2)/2 -0.5 + offset_range)
        set_seed_DiffAug(param)
        offset_y = np.random.randint(x.size(3)/2 - 0.5 - offset_range,high = x.size(3)/2 -0.5 + offset_range)

        if ((offset_x-x.size(2)/2+0.5)**2 + (offset_y-x.size(3)/2+0.5)**2) < (offset_range)**2 and offset_x%granularity1==0 and offset_y%granularity1==0:
            break
    

    
    radius_range = param.radius_range
    radius_min = param.radius_min
    thickness_range = param.thickness_range
    thickness_min = param.thickness_min

    #set_seed_DiffAug(param)
    #randradius_offset = np.random.randint(0, high = radius_range//granularity2)
    
    randradius_offset = 0

    #set_seed_DiffAug(param)
    #thickness_offset = np.random.randint(0, high = thickness_range//granularity2)
    thickness_offset = 0
    
    
    randradius_inside = radius_min + randradius_offset*granularity2
    randradius_outside = randradius_inside + thickness_min + thickness_offset*granularity2
   

    mask1,maskf1 = get_mask_circle(mask,maskf,length,randradius_inside,randradius_outside,offset_x,offset_y)
  
    #color= torch.sum(x*maskf1.unsqueeze(1), dim=(2,3), keepdim = True) / torch.sum(maskf1.unsqueeze(1), dim=(2,3), keepdim = True)

    return x*mask1.unsqueeze(1)  + color*maskf1.unsqueeze(1)
    #return x + fullcolor*maskf1.unsqueeze(1)



def get_mask_sector(mask,maskf,length,degree_beginn,degree_end,offset_x,offset_y):
   
    for i in range(length):
        for j in range(length):
            degree1 = np.arctan2(j-offset_y, i-offset_x) * 180 / np.pi + 180
            degree2 = np.arctan2(j-offset_y, i-offset_x) * 180 / np.pi + 360
            degree2 = np.mod(degree2,360)
            if degree_beginn < degree_end:
                if degree1 > degree_beginn and degree1 < degree_end:
                    mask[:,i,j] = 0
                    maskf[:,i,j] = 1
            else:
                if degree1 > degree_beginn or degree1 < degree_end:
                    mask[:,i,j] = 0
                    maskf[:,i,j] = 1
            
            if degree_beginn < degree_end:
                if degree2 > degree_beginn and degree2 < degree_end:
                    mask[:,i,j] = 0
                    maskf[:,i,j] = 1
            else:
                if degree2 > degree_beginn or degree2 < degree_end:
                    mask[:,i,j] = 0
                    maskf[:,i,j] = 1 
            


     

    return mask, maskf

def rand_nia3(x, param):
    
    length = x.size(2)
    

    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    maskf = torch.zeros(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    #color = torch.zeros(x.size(0),x.size(1), 1, 1, dtype=x.dtype, device=x.device)
    color = torch.rand(x.size(0), x.size(1), 1, 1, dtype=x.dtype, device=x.device)
    #noise = torch.rand(x.size(0), x.size(1), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    
    offset_range = param.offset_range
    set_seed_DiffAug(param)
    offset_x = np.random.randint(0 + offset_range,high = x.size(2) - offset_range)
    set_seed_DiffAug(param)
    offset_y = np.random.randint(0 + offset_range,high = x.size(3) - offset_range)
 
    degree_range = param.degree_range
    degree_min = param.degree_min
    degree_max = degree_min + degree_range

    if degree_range > 0:

        set_seed_DiffAug(param)
        degree_beginn = np.random.randint(0, high = 360)

        set_seed_DiffAug(param)
        degree_end = np.random.randint(degree_beginn+degree_min, high = degree_beginn+degree_max)
        degree_end = np.mod(degree_end,360)

        mask1,maskf1 = get_mask_sector(mask,maskf,length,degree_beginn,degree_end,offset_x,offset_y)
    else:
        mask1 = mask
        maskf1 = maskf

    #maskr2 = get_mask(mask,lengtt,ratio,randarea2)
    #maskr3 = get_mask(mask,lengtt,ratio,randarea3)

    #color= torch.sum(x*maskf1.unsqueeze(1), dim=(2,3), keepdim = True) / torch.sum(maskf1.unsqueeze(1), dim=(2,3), keepdim = True)

    return x*mask1.unsqueeze(1) + color*maskf1.unsqueeze(1)


def get_mask_rectangle(mask,maskf,length,degree,width,offset_x,offset_y):
    
    degree = np.mod(degree,180)
    
    # Parameters for calculation the distance 
    x = np.cos(degree)
    y = np.sin(degree)
    A = y
    B = -x
    C = 0
    
    for i in range(length):
        for j in range(length):      
            x = np.cos(degree)
            y = np.sin(degree)

            dist = np.abs(A*(i-offset_x) + B*(j-offset_y) + C) / (np.sqrt(A**2 + B**2))

            if dist < width/2 :
                mask[:,i,j] = 0
                maskf[:,i,j] = 1
            
    return mask, maskf

def rand_nia4(x, param):
    
    length = x.size(2)
    

    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    maskf = torch.zeros(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    #color = torch.zeros(x.size(0),x.size(1), 1, 1, dtype=x.dtype, device=x.device)
    color = torch.rand(x.size(0), x.size(1), 1, 1, dtype=x.dtype, device=x.device)
    #noise = torch.rand(x.size(0), x.size(1), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    
    offset_range = param.nia4_offset_range
    set_seed_DiffAug(param)
    offset_x = np.random.randint(0 + offset_range,high = x.size(2) - offset_range)
    set_seed_DiffAug(param)
    offset_y = np.random.randint(0 + offset_range,high = x.size(3) - offset_range)
 
    width_range = param.nia4_width_range
    width_min = param.nia4_width_min
    width_max = width_min + width_range
    
    if width_min < width_max:
        set_seed_DiffAug(param)
        width = np.random.randint(width_min, high = width_max)
    else:
        width = width_min
    
    set_seed_DiffAug(param)
    degree = np.random.randint(0, high = 180)


    mask1,maskf1 = get_mask_rectangle(mask,maskf,length,degree,width,offset_x,offset_y)
    

    #maskr2 = get_mask(mask,lengtt,ratio,randarea2)
    #maskr3 = get_mask(mask,lengtt,ratio,randarea3)

    #color= torch.sum(x*maskf1.unsqueeze(1), dim=(2,3), keepdim = True) / torch.sum(maskf1.unsqueeze(1), dim=(2,3), keepdim = True)

    #return x*mask1.unsqueeze(1) + color*maskf1.unsqueeze(1)
    return x + color*maskf1.unsqueeze(1)


def inarea(grid_x, grid_y, offset_x, offset_y, area_nr):

    if grid_x >= offset_x and grid_y >= offset_y:
        inar = 0
    
    if grid_x < offset_x and grid_y >= offset_y:
        inar = 1
    
    if grid_x < offset_x and grid_y < offset_y:
        inar = 2
    
    if grid_x >= offset_x and grid_y < offset_y:
        inar = 3
    
    if area_nr == inar: 
        output = True
    else:
        output = False 
    
    return output


def smooth(x,pic_num,grid_x,grid_y,smooth_threshold):

    size_x = x.size(2)
    size_y = x.size(3)
    differ = torch.zeros(x.size(1),dtype=x.dtype, device=x.device)
    
    output = False

    if grid_x == 0 or grid_x == size_x-1 or grid_y == 0 or grid_y == size_y-1:
       output = True 

    elif grid_x >= size_x/2 and grid_x >= size_y/2:
        for channel in range(x.size(1)):
            differ[channel] = abs(x[pic_num,channel,grid_x,grid_y] - x[pic_num,channel,grid_x+1,grid_y+1])
            +abs(x[pic_num,channel,grid_x,grid_y] - x[pic_num,channel,grid_x+1,grid_y])
            +abs(x[pic_num,channel,grid_x,grid_y] - x[pic_num,channel,grid_x,grid_y+1])
        differ_sum = torch.sum(differ)
        
        if differ_sum < smooth_threshold :
            output = True
        
    
    elif grid_x < size_x/2 and grid_y >= size_y/2:
        for channel in range(x.size(1)):
            differ[channel] = abs(x[pic_num,channel,grid_x,grid_y] - x[pic_num,channel,grid_x-1,grid_y+1])
            +abs(x[pic_num,channel,grid_x,grid_y] - x[pic_num,channel,grid_x-1,grid_y])
            +abs(x[pic_num,channel,grid_x,grid_y] - x[pic_num,channel,grid_x,grid_y+1])
        
        differ_sum = torch.sum(differ)
        
        if differ_sum < smooth_threshold :
            output = True
    
    elif grid_x < size_x/2 and grid_y < size_y/2:
        for channel in range(x.size(1)):
            differ[channel] = abs(x[pic_num,channel,grid_x,grid_y] - x[pic_num,channel,grid_x-1,grid_y-1])
            +abs(x[pic_num,channel,grid_x,grid_y] - x[pic_num,channel,grid_x-1,grid_y])
            +abs(x[pic_num,channel,grid_x,grid_y] - x[pic_num,channel,grid_x,grid_y-1])
        
        differ_sum = torch.sum(differ)
        
        if differ_sum < smooth_threshold :
            output = True
    
    elif grid_x >= size_x/2 and grid_y < size_y/2:
        for channel in range(x.size(1)):
            differ[channel] = abs(x[pic_num,channel,grid_x,grid_y] - x[pic_num,channel,grid_x+1,grid_y+1])
            +abs(x[pic_num,channel,grid_x,grid_y] - x[pic_num,channel,grid_x-1,grid_y])
            +abs(x[pic_num,channel,grid_x,grid_y] - x[pic_num,channel,grid_x,grid_y-1])
        
        differ_sum = torch.sum(differ)
        
        if differ_sum < smooth_threshold :
            output = True
    
    
    return output

def convert_to_xy(layer, num_layer, pixel, num_pixel):
    
    if pixel //(num_pixel/4) == 0:
        case = 0
        x = pixel%(num_pixel/4) + layer
        y = layer
    elif pixel // (num_pixel/4) == 1:
        case = 1
        x = num_layer*2 - layer -1
        y = layer + pixel%(num_pixel/4)
    elif pixel // (num_pixel/4) == 2:
        case = 2
        x = num_layer*2 - pixel%(num_pixel/4) - layer -1
        y = num_layer*2 - layer -1
    elif pixel // (num_pixel/4) == 3:
        case = 3
        x = layer 
        y = num_layer*2 - pixel%(num_pixel/4) - layer -1


    return int(x),int(y)

def get_frontier(grid_x, grid_y, pixel, num_pixel):

    if pixel//(num_pixel/4) == 0:
        case = 0
        if pixel%(num_pixel/4) == 0:
            x = grid_x + 1
            y = grid_y + 1
        else: 
            x = grid_x 
            y = grid_y + 1

    elif pixel//(num_pixel/4) == 1:
        case = 1
        if pixel%(num_pixel/4) == 0:
            x = grid_x - 1
            y = grid_y + 1
        else: 
            x = grid_x - 1
            y = grid_y 

    elif pixel//(num_pixel/4) == 2:
        case = 2
        if pixel%(num_pixel/4) == 0:
            x = grid_x - 1
            y = grid_y - 1
        else: 
            x = grid_x 
            y = grid_y - 1

    elif pixel//(num_pixel/4) == 3:
        case = 3
        if pixel%(num_pixel/4) == 0:
            x = grid_x + 1
            y = grid_y - 1
        else: 
            x = grid_x + 1
            y = grid_y 
   
    return int(x),int(y),int(x),int(y),int(x),int(y)
        


'''
def rand_circlecutout(x, param):
    
    
    ring = param.ring_circlecutout
    
    
    radius = param.radius_circlecutout
    thickness = param.thickness_circlecutout
    margin = param.margin_circlecutout
    smooth_threshold = 0.7
    
    mosaic_size = param.mosaicsize_circlecutout
    
    offset_x = x.size(2) /2
    offset_y = x.size(3) /2
    
    
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2), size=[1, 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3), size=[1, 1, 1], device=x.device)
    
    set_seed_DiffAug(param)
    r_max = torch.randint(3, 8, size=[1, 1, 1], device=x.device)
    
  
    #radius_min = np.random.randint(low = x.size(2)//8, high = x.size(2)//2, size=[x.size(0)])
    #radius_min = np.random.randint(low = x.size(2)//8, high = x.size(2)//2, size=[x.size(0)])
    #thickness  = np.random.randint(low = 0, high = 3, size=[x.size(0)])
    #radius_max = radius_min + thickness
    
    set_seed_DiffAug(param)
    area_nr = torch.randint(0, 6, size=[1], device=x.device)

    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    noise = torch.rand(x.size(0), x.size(1), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    fillmask = torch.zeros(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    #mosaic = torch.zeros(x.size(0), x.size(1), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    #onecolor = torch.zeros(x.size(0), x.size(1), x.size(2), x.size(3), dtype=x.dtype, device=x.device)

    color_accumulated = torch.zeros(x.size(0), x.size(1),1,1,dtype=x.dtype, device=x.device)
    pixel_count = torch.zeros(x.size(0),1,1,1,dtype=x.dtype, device=x.device)  
    frontier = torch.zeros(x.size(0),x.size(2),x.size(3),dtype=x.dtype, device=x.device)  
    smo = torch.zeros(x.size(0),x.size(2),x.size(3),dtype=x.dtype, device=x.device)  
    
    #for pic in range(x.size(0)):
    #r_min = radius_min[pic]
    #r_max = radius_max[pic] 
    #radius_max = radius + ring*thickness + margin
    #radius_min = 0 if ring == 0 else radius + (ring-1)*thickness - margin

    #inarea(grid_x, grid_y, offset_x, offset_y, area_nr)
    #print('area_nr', area_nr)
    for pic_num in range(x.size(0)):
        for grid_x in range(x.size(2)):
            for grid_y in range(x.size(3)):
                if smooth(x,pic_num,grid_x,grid_y,smooth_threshold) and (((grid_x- offset_x)**2 + abs(grid_y- offset_y)**2) >= 196):
                    mask[pic_num, grid_x, grid_y] = 0
                    fillmask[pic_num, grid_x, grid_y] = 1
                    #and inarea(grid_x, grid_y, offset_x, offset_y, area_nr)
                    #mosaic[:, :, grid_x, grid_y] = x[:, :, (grid_x//mosaic_size)*mosaic_size, (grid_y//mosaic_size)*mosaic_size]
                    #onecolor[pic_num, :, grid_x, grid_y] = 1
                    pixel_count[pic_num,0,0,0] = pixel_count[pic_num,0,0,0] + 1
                    color_accumulated[pic_num,:,0,0] = color_accumulated[pic_num,:,0,0] + x[pic_num, :, grid_x, grid_y]

                else:
                    frontier[pic_num,grid_x,grid_y] = 1

    for pic_num in range(x.size(0)):
        for channel in range(x.size(1)):
            color_accumulated[pic_num,channel,0,0] = color_accumulated[pic_num,channel,0,0] / pixel_count[pic_num,0,0,0]
    
    
    #colormask = onecolor * color_accumulated 
    
    x = x* mask.unsqueeze(1) 
    
    return x
'''

def rand_circlecutout(x, param):
    
    '''
    ring = param.ring_circlecutout
    '''

    radius = param.radius_circlecutout
    thickness = param.thickness_circlecutout
    margin = param.margin_circlecutout
    smooth_threshold = 0.7
    
    mosaic_size = param.mosaicsize_circlecutout
    
    offset_x = x.size(2) /2
    offset_y = x.size(3) /2
    
    '''
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2), size=[1, 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3), size=[1, 1, 1], device=x.device)
    
    set_seed_DiffAug(param)
    r_max = torch.randint(3, 8, size=[1, 1, 1], device=x.device)
    '''
  
    #radius_min = np.random.randint(low = x.size(2)//8, high = x.size(2)//2, size=[x.size(0)])
    #radius_min = np.random.randint(low = x.size(2)//8, high = x.size(2)//2, size=[x.size(0)])
    #thickness  = np.random.randint(low = 0, high = 3, size=[x.size(0)])
    #radius_max = radius_min + thickness
    
    set_seed_DiffAug(param)
    area_nr = torch.randint(0, 6, size=[1], device=x.device)

    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    noise = torch.rand(x.size(0), x.size(1), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    fillmask = torch.zeros(x.size(0), x.size(1), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    #mosaic = torch.zeros(x.size(0), x.size(1), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    #onecolor = torch.zeros(x.size(0), x.size(1), x.size(2), x.size(3), dtype=x.dtype, device=x.device)

    color_accumulated = torch.zeros(x.size(0), x.size(1),1,1,dtype=x.dtype, device=x.device)
    pixel_count = torch.zeros(x.size(0),1,1,1,dtype=x.dtype, device=x.device)  
    frontier = torch.zeros(x.size(0),x.size(2),x.size(3),dtype=x.dtype, device=x.device)  
    smo = torch.zeros(x.size(0),x.size(2),x.size(3),dtype=x.dtype, device=x.device)  
    
    #for pic in range(x.size(0)):
    #r_min = radius_min[pic]
    #r_max = radius_max[pic] 
    #radius_max = radius + ring*thickness + margin
    #radius_min = 0 if ring == 0 else radius + (ring-1)*thickness - margin

    #inarea(grid_x, grid_y, offset_x, offset_y, area_nr)
    #print('area_nr', area_nr)
    num_layer = x.size(2)//2
    

    for pic_num in range(x.size(0)):
        for layer in range(num_layer):
            num_pixel = (num_layer - layer)*8 - 4
            for pixel in range(num_pixel):
                grid_x, grid_y = convert_to_xy(layer, num_layer, pixel, num_pixel)
                
                if smooth(x,pic_num,grid_x,grid_y,smooth_threshold) and frontier[pic_num,grid_x,grid_y] == 0:
                    mask[pic_num, grid_x, grid_y] = 0
                    fillmask[pic_num,:, grid_x, grid_y] = 1
                    #and inarea(grid_x, grid_y, offset_x, offset_y, area_nr)
                    #mosaic[:, :, grid_x, grid_y] = x[:, :, (grid_x//mosaic_size)*mosaic_size, (grid_y//mosaic_size)*mosaic_size]
                    #onecolor[pic_num, :, grid_x, grid_y] = 1
                    pixel_count[pic_num,0,0,0] = pixel_count[pic_num,0,0,0] + 1
                    color_accumulated[pic_num,:,0,0] = color_accumulated[pic_num,:,0,0] + x[pic_num, :, grid_x, grid_y]

                else:
                    frontier_x1, frontier_y1, frontier_x2, frontier_y2, frontier_x3, frontier_y3 = get_frontier(grid_x, grid_y, pixel, num_pixel)
                    frontier[pic_num,frontier_x1,frontier_y1] = 1
                    frontier[pic_num,frontier_x2,frontier_y2] = 1
                    frontier[pic_num,frontier_x3,frontier_y3] = 1

    for pic_num in range(x.size(0)):
        for channel in range(x.size(1)):
            color_accumulated[pic_num,channel,0,0] = color_accumulated[pic_num,channel,0,0] / pixel_count[pic_num,0,0,0]
    
    
    #colormask = onecolor * color_accumulated 
    
    x = x* mask.unsqueeze(1) +  color_accumulated * fillmask
    
    return x

def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randb[:] = randb[0].clone()
    x = x + (randb - 0.5)*ratio
    return x 

def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        rands[:] = rands[0].clone()
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randc[:] = randc[0].clone()
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        translation_x[:] = translation_x[0].clone()
        translation_y[:] = translation_y[0].clone()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x

def rand_erasing(x, param):
    # A random part of the image is erased, not well developed yet :=)
    ratio = param.erasing
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        offset_x[:] = offset_x[0].clone()
        offset_y[:] = offset_y[0].clone()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x

def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        offset_x[:] = offset_x[0].clone()
        offset_y[:] = offset_y[0].clone()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x

def autoaug(x, param):
   
    x = custom_aug(x, param)
    return x

def randaug(x, param):
   
    x = custom_aug(x, param)
    return x

def imagenetaug(x, param):
    
    x = custom_aug(x, param)
    return x

def cifaraug(x, param):
   
    x = custom_aug(x, param)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'brightness': [rand_brightness],
    'saturation': [rand_saturation],
    'contrast': [rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
    'autoaug': [autoaug],
    'randaug': [randaug],
    'imagenetaug': [imagenetaug],
    'cifaraug': [cifaraug],
    'new': [rand_new],
    'new1': [rand_new1],
    'rotatedegree': [rand_rotatedegree],
    'nia1': [rand_nia1],
    'nia2': [rand_nia2],
    'nia3': [rand_nia3],
    'nia4': [rand_nia4],
    'nia6': [rand_nia6],
    'circlecutout': [rand_circlecutout],
}
