import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
import wandb
os.environ["WANDB_SILENT"] = "true"
#os.environ['WANDB_MODE'] = 'offline'



def main(strategy_1,strategy_2,parameter_1,parameter_2):

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='TWO', help='DC/DSA/TWO')
    parser.add_argument('--stage', type=str, default='two', help='one stage/two stage')
    parser.add_argument('--evaluate_synthetic_data', type=str, default='True', help='True/False')
    parser.add_argument('--create_synthetic_data', type=str, default='True', help='True/False')
    parser.add_argument('--data_path_of_synthetic_data', type=str, default='data', help='dataset path for getting synthetic data')
    parser.add_argument('--seedsetting', type=str, default=parameter_1, help='seed = ?')
    parser.add_argument('--dataset', type=str, default=parameter_2, help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_eval', type=int, default=5, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--normalize_data', action="store_true", default=True, help='the number of evaluating randomly initialized models')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    #We use --dsa_strategy_stage_aug and set --dsa_strategy_stage_1 to mute it.
    parser.add_argument('--dsa_strategy_stage_aug', type=str, default=strategy_1, help="'autoaug','randaug','imagenet_aug','cifar_aug'")
    parser.add_argument('--dsa_strategy_stage_1', type=str, default='none', help='differentiable Siamese augmentation strategy, full = color_crop_cutout_flip_scale_rotate')
    
    parser.add_argument('--dsa_strategy_stage_2', type=str, default=strategy_2, help='differentiable Siamese augmentation strategy, full = color_crop_cutout_flip_scale_rotate')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result/%s'%(parameter_2), help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')

    args = parser.parse_args() 
    args.outer_loop, args.inner_loop = get_loops(args.ipc) 
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param_stage_1 = ParamDiffAug()
    args.dsa_param_stage_2 = ParamDiffAug()
    
    args.dsa = True if args.method == 'DSA' or 'TWO' else False 
    args.twostage = True if args.method == 'TWO' else False
   
    args.evaluate_sd = True if args.evaluate_synthetic_data == 'True' else False
    args.create_sd = True if args.create_synthetic_data == 'True' else False
    
    args.dsa_param_stage_2.ring_circlecutout = 0
    args.dsa_param_stage_2.radius_circlecutout = 5
    args.dsa_param_stage_2.thickness_circlecutout = 8

    Threshold_Random_DA = 1.0 # DA were applied in parameter_1% cases

    
    
    wandb.init(project= "Creation_1000")
    wandb.config.method = args.method
    wandb.config.stage = args.stage
    wandb.config.seedsetting = args.seedsetting
    wandb.config.dataset = args.dataset
    wandb.config.model = args.model
    wandb.config.ipc = args.ipc
    #wandb.config.Threshold_Random_DA = Threshold_Random_DA
    wandb.config.dsa_strategy_stage_1 = args.dsa_strategy_stage_aug
    wandb.config.dsa_strategy_stage_2 = args.dsa_strategy_stage_2
    
    wandb.config.radius_circlecutout = args.dsa_param_stage_2.radius_circlecutout 
    wandb.config.thickness_circlecutout = args.dsa_param_stage_2.thickness_circlecutout 


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
        if args.dsa_strategy_stage_aug == 'autoaug':
            if args.dataset == 'tinyimagenet':
                data_transforms = transforms.Compose([transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.IMAGENET)])
            elif args.dataset == 'CIFAR10':
                data_transforms = transforms.Compose([transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10)])
            elif args.dataset == 'SVHN':
                data_transforms = transforms.Compose([transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.SVHN)])
            else:
                exit('unknown augmentation method: %s'%args.dsa_strategy_stage_aug)

        elif args.dsa_strategy_stage_aug == 'randaug':
            data_transforms = transforms.Compose([transforms.RandAugment(num_ops=1)])
        elif args.dsa_strategy_stage_aug == 'imagenetaug':
            data_transforms = transforms.Compose([transforms.RandomCrop(size, padding=4), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)])
        elif args.dsa_strategy_stage_aug == 'cifaraug':
            data_transforms = transforms.Compose([transforms.RandomCrop(size, padding=4), transforms.RandomHorizontalFlip()])
        else:
            exit('unknown augmentation method: %s'%args.dsa_strategy_stage_aug)
        normalized_d = data_transforms(normalized_d.to(torch.uint8))
        normalized_d = normalized_d / 255.0

        # print("changes after autoaug: ", (normalized_d - image_syn_vis).pow(2).sum().item())

        if args.normalize_data:
            for ch in range(channel):
                normalized_d[:, ch] = (normalized_d[:, ch] - mean[ch])  / std[ch]

        if args.dsa_strategy_stage_aug == 'cifaraug':
            cutout_transform = transforms.Compose([Cutout(16, 1)])
            normalized_d = cutout_transform(normalized_d)

        return normalized_d

    

    seed = args.seedsetting*100000

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    eval_it_pool = np.arange(0, args.Iteration+1, 1000).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    


    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []
    


    
    print('\n================== Exp %d ==================\n ')
    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)
    
    if args.create_sd:
        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        

        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
        else:
            print('initialize synthetic data from random noise')

    if args.create_sd == False: 
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]


    ''' training '''
    optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
    optimizer_img.zero_grad()
    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins'%get_time())

    for it in range(args.Iteration+1):

        ''' Evaluate synthetic data '''
        if it in eval_it_pool and args.evaluate_sd:
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                if args.dsa:
                    args.epoch_eval_train = 1000
                    args.dc_aug_param = None
                    print('DSA augmentation strategy: \n', args.dsa_strategy_stage_1)
                    print('DSA augmentation parameters: \n', args.dsa_param_stage_1.__dict__)
                else:
                    args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                    print('DC augmentation parameters: \n', args.dc_aug_param)

                if args.dsa or args.dc_aug_param['strategy'] != 'none':
                    if args.twostage:
                        args.epoch_eval_train = 1000  # Training with two data augmentation needs way more epochs.
                    else:
                        args.epoch_eval_train = 1000  # Training with data augmentation needs more epochs.
                else:
                    args.epoch_eval_train = 300

                accs = []
                for it_eval in range(args.num_eval):
                    seed = seed +1 
                    net_eval = get_network(model_eval, seed, channel, num_classes, im_size).to(args.device) # get a random model
                    image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                    image_syn_eval_DA_total = image_syn_eval
                    label_syn_eval_DA_total = label_syn_eval

                    _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval_DA_total, label_syn_eval_DA_total, testloader, args, stage = 2)
                    accs.append(acc_test)
                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
                wandb.log({'evaluation_acc_mean': np.mean(accs)})
                wandb.log({'evaluation_acc_std': np.std(accs)})

                if it == args.Iteration: # record the final results
                    accs_all_exps[model_eval] += accs

            ''' visualize and save '''
            
            save_name = os.path.join(args.save_path, 'vis_syn_%s_strategy_stage_1=%s_%s_%s_%dipc_Seed=%s_iter%d.png'%(args.method, args.dsa_strategy_stage_aug, args.dataset, args.model, args.ipc,parameter_2, it))
            image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
            for ch in range(channel):
                image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
            image_syn_vis[image_syn_vis<0] = 0.0
            image_syn_vis[image_syn_vis>1] = 1.0
            save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.

        if args.create_sd:
            ''' Train synthetic data '''
            seed = seed +1
            net = get_network(args.model, seed, channel, num_classes, im_size).to(args.device) # get a random model
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            loss_avg = 0
            args.dc_aug_param = None  # Mute the DC augmentation when learning synthetic data (in inner-loop epoch function) in oder to be consistent with DC paper.


            for ol in range(args.outer_loop):

                ''' freeze the running mu and sigma for BatchNorm layers '''
                # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                # This would make the training with BatchNorm layers easier.

                BN_flag = False
                BNSizePC = 16  # for batch normalization
                for module in net.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True
                if BN_flag:
                    img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                    net.train() # for updating the mu, sigma of BatchNorm
                    output_real = net(img_real) # get running mu, sigma
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer


                ''' update synthetic data '''
                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    
                    #img_real = get_images(c, args.batch_real)
                    #lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c

                    

 

                    seed = seed + 1
                    Random_DA = np.random.random() 
                    

                    if args.twostage and Random_DA < Threshold_Random_DA:  # we add DAs on the original Data
                        seed = seed + 1
                        img_real_load = get_images(c, args.batch_real)
                        img_real = custom_aug(img_real_load, args)
                        lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                    else:
                        img_real = get_images(c, args.batch_real)
                        lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                        
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                    lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c

                   
                        
                    output_real = net(img_real)
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    output_syn = net(img_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                    loss += match_loss(gw_syn, gw_real, args)

                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
                loss_avg += loss.item()

                if ol == args.outer_loop - 1:
                    break


                ''' update network '''
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
                for il in range(args.inner_loop):
                    epoch('train', trainloader, net, optimizer_net, criterion, args, aug = True if args.dsa else False, stage = 1)


            loss_avg /= (num_classes*args.outer_loop)

            if it%10 == 0:
                print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))
                wandb.log({'creation_loss': loss_avg})
            
                

            if it == args.Iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': copy.deepcopy(image_syn.detach().cpu())}, os.path.join(args.save_path, 'image_syn_%s_strategy_stage_1=%s_%s_%s_%dipc_Seed=%s.pt'%(args.method, args.dsa_strategy_stage_aug, args.dataset, args.model, args.ipc,parameter_1)))
                torch.save({'data': copy.deepcopy(label_syn.detach().cpu())}, os.path.join(args.save_path, 'label_syn_%s_strategy_stage_1=%s_%s_%s_%dipc_Seed=%s.pt'%(args.method, args.dsa_strategy_stage_aug, args.dataset, args.model, args.ipc,parameter_1)))
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_syn_%s_strategy_stage_1=%s_%s_%s_%dipc_Seed=%s.pt'%(args.method, args.dsa_strategy_stage_aug, args.dataset, args.model, args.ipc,parameter_1)))
    
    


    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiment, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(1, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))
    wandb.finish()




Test_list = {
    'strategy_1': ['autoaug'], #'autoaug'(only for CIFAR10 and SVHN),'randaug'
    'strategy_2': ['none'], #'none','color','crop','cutout','flip','scale','rotate','color_crop_cutout_flip_scale_rotate','autoaug'(only for CIFAR10 and SVHN),'randaug'
    'parameter_1': [1], #seedsetting
    'parameter_2': ['CIFAR10'], #Dataset: Dataset 'CIFAR10','MNIST','SVHN' 
}


if __name__ == '__main__':
    for stra_1 in Test_list['strategy_1']:
        for stra_2 in Test_list['strategy_2']:
            for parameter_1 in Test_list['parameter_1']:
                for parameter_2 in Test_list['parameter_2']:     
                    main(stra_1,stra_2,parameter_1,parameter_2) 



