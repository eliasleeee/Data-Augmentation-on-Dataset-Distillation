import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
import wandb
os.environ["WANDB_SILENT"] = "true"
#os.environ['WANDB_MODE'] = 'offline'


def main(para_1,para_2,parameter_1,parameter_2):

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='TWO', help='DC/DSA/')
    parser.add_argument('--stage', type=str, default='two', help='one stage/two stage')
    parser.add_argument('--evaluate_synthetic_data', type=str, default='True', help='True/False')
    parser.add_argument('--create_synthetic_data', type=str, default='True', help='True/False')
    parser.add_argument('--data_path_of_synthetic_data', type=str, default='data', help='dataset path for getting synthetic data')
    parser.add_argument('--seedsetting', type=str, default=2, help='seed = ?')
    parser.add_argument('--dataset', type=str, default=parameter_2, help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_eval', type=int, default=10, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=10, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--normalize_data', action="store_true", default=True, help='the number of evaluating randomly initialized models')
    parser.add_argument('--dsa_strategy_stage_1', type=str, default=para_1, help='differentiable Siamese augmentation strategy, full = color_crop_cutout_flip_scale_rotate')
    parser.add_argument('--dsa_strategy_stage_2', type=str, default=para_2, help='differentiable Siamese augmentation strategy, full = color_crop_cutout_flip_scale_rotate,autoaug,randaug,imagenet_aug,cifar_aug')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--sys_data_path', type=str, default='result/%s'%(parameter_2), help='synthetic dataset path')
    parser.add_argument('--save_path', type=str, default='result/stage1_none', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')

    args = parser.parse_args()
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param_stage_1 = ParamDiffAug()
    args.dsa_param_stage_2 = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' or 'TWO' else False
    args.twostage = True if args.stage == 'two' else False
    args.evaluate_sd = True if args.evaluate_synthetic_data == 'True' else False
    args.create_sd = True if args.create_synthetic_data == 'True' else False
    args.dsa_param_stage_2.ring_circlecutout = 0
    args.dsa_param_stage_2.radius_circlecutout = 5
    args.dsa_param_stage_2.thickness_circlecutout = 5
    args.dsa_param_stage_2.margin_circlecutout = 2
    args.dsa_param_stage_2.mosaicsize_circlecutout = 2


    args.dsa_param_stage_2.offset_range_nia2 = 50
    args.dsa_param_stage_2.radius_min = 32
    args.dsa_param_stage_2.radius_range = 12
    args.dsa_param_stage_2.thickness_min = 2
    args.dsa_param_stage_2.thickness_range = 8
    
    args.aug = args.dsa_strategy_stage_2
    args.dsa_param_stage_2.normalize_data = args.normalize_data
    args.dsa_param_stage_2.dataset = args.dataset
    args.dsa_param_stage_2.aug = args.aug

    
    # args.dsa_param_stage_2.offset_range = para_0
    # args.dsa_param_stage_2.degree_min = para_1
    # args.dsa_param_stage_2.degree_range = para_2


    # args.dsa_param_stage_2.nia4_offset_range = para_0
    # args.dsa_param_stage_2.nia4_width_range = para_1
    # args.dsa_param_stage_2.nia4_width_min = para_2

    num_eval_epoch = 1000

    
    wandb.init(project= "Evaluation")
    wandb.config.method = args.method
    wandb.config.dsa_strategy_stage_1 = args.dsa_strategy_stage_1
    wandb.config.dsa_strategy_stage_2 = args.dsa_strategy_stage_2
    wandb.config.Threshold_Random_DA = args.Threshold_Random_DA
    wandb.config.seedsetting = parameter_1
    wandb.config.dataset = args.dataset
    wandb.config.model = args.model
    wandb.config.ipc = args.ipc
    wandb.config.num_eval_epoch = num_eval_epoch
    
    # wandb.config.offset_range = args.dsa_param_stage_2.offset_range
    # wandb.config.degree_min = args.dsa_param_stage_2.degree_min
    # wandb.config.degree_range = args.dsa_param_stage_2.degree_range

    # wandb.config.nia4_offset_range = args.dsa_param_stage_2.nia4_offset_range 
    # wandb.config.nia4_width_range = args.dsa_param_stage_2.nia4_width_range 
    # wandb.config.nia4_width_min = args.dsa_param_stage_2.nia4_width_min 

    
    #wandb.config.offset_range = args.dsa_param_stage_2.offset_range_nia2
    
    #wandb.config.radius_min = args.dsa_param_stage_2.radius_min
    #wandb.config.radius_range = args.dsa_param_stage_2.radius_range
    #wandb.config.thickness_min = args.dsa_param_stage_2.thickness_min
    #wandb.config.thickness_range = args.dsa_param_stage_2.thickness_range

    # wandb.config.radius_circlecutout = args.dsa_param_stage_2.radius_circlecutout 
    # wandb.config.thickness_circlecutout = args.dsa_param_stage_2.thickness_circlecutout
    # wandb.config.margin_circlecutout = args.dsa_param_stage_2.margin_circlecutout 
    # wandb.config.mosaicsize_circlecutout = args.dsa_param_stage_2.mosaicsize_circlecutout
    


    seed = args.seedsetting*100000

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)


    #data_image = torch.load(os.path.join(args.sys_data_path, 'image_syn_%s_%s_%s_%dipc.pt'%(args.dsa_strategy_stage_1, args.dataset, args.model, args.ipc)))
    #data_label = torch.load(os.path.join(args.sys_data_path, 'label_syn_%s_%s_%s_%dipc.pt'%(args.dsa_strategy_stage_1, args.dataset, args.model, args.ipc)))
    #data_image = torch.load(os.path.join(args.sys_data_path, 'image_syn_none_CIFAR10_ConvNet_10ipc.pt'))
    #data_label = torch.load(os.path.join(args.sys_data_path, 'label_syn_none_CIFAR10_ConvNet_10ipc.pt'))
 
    data_image = torch.load(os.path.join(args.sys_data_path, 'image_syn_%s_strategy_stage_1=%s_%s_%s_%dipc_Seed=%s.pt'%(args.method, args.dsa_strategy_stage_1, args.dataset, args.model, args.ipc,parameter_1)))
    data_label = torch.load(os.path.join(args.sys_data_path, 'label_syn_%s_strategy_stage_1=%s_%s_%s_%dipc_Seed=%s.pt'%(args.method, args.dsa_strategy_stage_1, args.dataset, args.model, args.ipc,parameter_1)))
    """data_image = torch.load(os.path.join(args.sys_data_path, 'image_syn_%s_strategy_stage_0=%s_%s_%s_%dipc.pt'%(args.method, args.dsa_strategy_stage_0, args.dataset, args.model, args.ipc)))
    data_label = torch.load(os.path.join(args.sys_data_path, 'label_syn_%s_strategy_stage_0=%s_%s_%s_%dipc.pt'%(args.method, args.dsa_strategy_stage_0, args.dataset, args.model, args.ipc)))"""
    #training_data = dc_data['data']
    #image_syn, label_syn = training_data[-1]
    

    image_syn  = data_image['data']
    label_syn  = data_label['data']
    
    

    """
    data_image = torch.load(os.path.join(args.sys_data_path, 'image_syn_%s_%s_%s_%dipc.pt'%(args.dsa_strategy_stage_1, args.dataset, args.model, args.ipc)))
    data_label = torch.load(os.path.join(args.sys_data_path, 'label_syn_%s_%s_%s_%dipc.pt'%(args.dsa_strategy_stage_1, args.dataset, args.model, args.ipc)))
   
    image_syn  = data_image['data']
    label_syn  = data_label['data']
    
    image_syn = torch.load(os.path.join(args.sys_data_path, 'images_best.pt'))
    label_syn = torch.load(os.path.join(args.sys_data_path, 'labels_best.pt'))

    """


    
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    


    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    
    print('\n================== Exp %d ==================\n ')
    
    
    if args.evaluate_sd:
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


        
        for model_eval in model_eval_pool:
            print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s'%(args.model, model_eval))
            if args.dsa:
                args.epoch_eval_train = 1000
                args.dc_aug_param = None
                print('DSA augmentation strategy: \n', args.dsa_strategy_stage_1)
                print('DSA augmentation parameters: \n', args.dsa_param_stage_1.__dict__)
            else:
                args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                print('DC augmentation parameters: \n', args.dc_aug_param)

            if args.dsa or args.dc_aug_param['strategy'] != 'none':
                
                args.epoch_eval_train = num_eval_epoch  # Training with data augmentation needs more epochs.
            else:
                args.epoch_eval_train = 300

            accs = []
            for it_eval in range(args.num_eval):
                seed = seed +1 
                net_eval = get_network(model_eval, seed, channel, num_classes, im_size).to(args.device) # get a random model
                image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                image_syn_eval_DA_total = image_syn_eval
                label_syn_eval_DA_total = label_syn_eval
                
                '''
                #args.dsa_param_stage_2.ring_circlecutout = 0
                
                print('it_eval',it_eval)
                print('args.dsa_strategy_stage_2',args.dsa_strategy_stage_2)
                #image_syn_eval_DA = DiffAugment(image_syn_eval, args.dsa_strategy_stage_2, seed=seed, param=args.dsa_param_stage_2)
                #image_syn_eval_DA_total = image_syn_eval_DA
                #label_syn_eval_DA_total = label_syn_eval
                
                
                
                if args.twostage: # in case of two stage Data Augmentation, we also add the DA here
                    for DA in range(3): # we add 1 different random DAs here
                        print('DA',DA)
                        seed = seed +1
                        args.dsa_param_stage_2.ring_circlecutout = DA+1
                        image_syn_eval_DA = DiffAugment(image_syn_eval, args.dsa_strategy_stage_2, seed=seed, param=args.dsa_param_stage_2)
                        image_syn_eval_DA_total = torch.cat((image_syn_eval_DA_total,image_syn_eval_DA), dim=0) 
                        label_syn_eval_DA_total = torch.cat((label_syn_eval_DA_total,label_syn_eval), dim=0)
                        print('DA1',DA)
                '''

                
                _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, stage = 2)
                print('train')
                accs.append(acc_test)
            print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
            wandb.log({'evaluation_acc_mean': np.mean(accs)})
            wandb.log({'evaluation_acc_std': np.std(accs)})

            accs_all_exps[model_eval] += accs
    wandb.finish()

 
'''
Test_list = {
    'strategy_0': ['none'], #'none','color','crop','cutout','flip','scale','rotate'
    'strategy_1': ['none'], #'none','color','crop','cutout','flip','scale','rotate'
    'strategy_2': ['cutout'], #'none','color','crop','cutout','flip','scale','rotate','new','new1','new2'
}


Test_list = {
    'offset_range': [1,3,5], #'none','color','crop','cutout','flip','scale','rotate'
    'radius_min': [1,3,5,7,9,11,13,15], #'none','color','crop','cutout','flip','scale','rotate'
    'radius_range': [2,4,6], #'none','color','crop','cutout','flip','scale','rotate','new','new1','new2'
    'thickness_min': [0,2,4,6], #'none','color','crop','cutout','flip','scale','rotate'
    'thickness_range': [2,4,6,8], #'none','color','crop','cutout','flip','scale','rotate','new','new1','new2'
}


Test_list = {
    'offset_range': [50], # degree_min 'none','color','crop','cutout','flip','scale','rotate' 
    'nia2_radius_min': [32], # degree_range 'none','color','crop','cutout','flip','scale','rotate'
    'nia2_radius_range': [12], #'none','color','crop','cutout','flip','scale','rotate','new','new1','new2'
    'thickness_min': [2], #'none','color','crop','cutout','flip','scale','rotate'
    'thickness_range': [8], #'none','color','crop','cutout','flip','scale','rotate','new','new1','new2'
}
'''

Test_list = {
    'strategy_1': ['none'], ##'none','color','crop','cutout','flip','scale','rotate','color_crop_cutout_flip_scale_rotate','autoaug'(only for CIFAR10 and SVHN),'randaug'
    'strategy_2': ['none'], ##'none','color','crop','cutout','flip','scale','rotate','color_crop_cutout_flip_scale_rotate','autoaug'(only for CIFAR10 and SVHN),'randaug'
    'parameter_1': [1], # Seedsetting: 1,2,3
    'parameter_2': ['CIFAR10'], # Dataset: MNIST,CIFAR10,SVHN
} 

   



if __name__ == '__main__':
        for para_1 in Test_list['strategy_1']:
            for para_2 in Test_list['strategy_2']:
                for parameter_1 in Test_list['parameter_1']:
                    for parameter_2 in Test_list['parameter_2']:
                            main(para_1,para_2,parameter_1,parameter_2) 