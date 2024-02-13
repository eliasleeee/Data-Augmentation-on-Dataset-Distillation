### Setup
install packages in the requirements.

###  Effect of data augmentation applied on Stage I - Table 1
```
python main_creation_Tradition.py   
#--model: ConvNet 
#--dataset: MNIST, SVHN, CIFAR10
#--seedsetting: 1,2,3
#--ipc (images/class): 10
#--Iteration: 1000
#--dsa_strategy_stage_1: 'none','color','crop','cutout','flip','scale','rotate','color_crop_cutout_flip_scale_rotate'
#--dsa_strategy_stage_2: 'none'

python main_creation_AutoAug.py   
#--model: ConvNet 
#--dataset: MNIST, SVHN, CIFAR10
#--seedsetting: 1,2,3
#--ipc (images/class): 10
#--Iteration: 1000
#--dsa_strategy_stage_1: 'autoaug'(only for CIFAR10 and SVHN),'randaug'
#--dsa_strategy_stage_2: 'none'

```


###  Effect of data augmentation applied on Stage II - Table 2
```
python main_creation_Tradition.py   
#--model: ConvNet 
#--dataset: MNIST, SVHN, CIFAR10
#--seedsetting: 1,2,3
#--ipc (images/class): 10
#--Iteration: 1000
#--dsa_strategy_stage_1: 'none'
#--dsa_strategy_stage_2: 'none','color','crop','cutout','flip','scale','rotate','color_crop_cutout_flip_scale_rotate','autoaug'(only for CIFAR10 and SVHN),'randaug'

```



###  The influence of Data Augmentation applied in Stage I on Stage II
```
python main_creation_Tradition.py   
#--model: ConvNet 
#--dataset: MNIST, SVHN, CIFAR10
#--seedsetting: 1,2,3
#--ipc (images/class): 10
#--Iteration: 1000
#--dsa_strategy_stage_1: 'none','color','crop','cutout','flip','scale','rotate','color_crop_cutout_flip_scale_rotate'
#--dsa_strategy_stage_2: 'none','color','crop','cutout','flip','scale','rotate','color_crop_cutout_flip_scale_rotate','autoaug'(only for CIFAR10 and SVHN),'randaug'

python main_creation_AutoAug.py   
#--model: ConvNet 
#--dataset: MNIST, SVHN, CIFAR10
#--seedsetting: 1,2,3
#--ipc (images/class): 10
#--Iteration: 1000
#--dsa_strategy_stage_1: 'autoaug'(only for CIFAR10 and SVHN),'randaug'
#--dsa_strategy_stage_2: 'none','color','crop','cutout','flip','scale','rotate','color_crop_cutout_flip_scale_rotate','autoaug'(only for CIFAR10 and SVHN),'randaug'

```


### Recording of updates on synthetic image using standard DC
```
python main_creation_Tradition.py   
#--model: ConvNet 
#--dataset:CIFAR10
#--seedsetting: 1,2,3
#--ipc (images/class): 10
#--Iteration: 1000
#--dsa_strategy_stage_1: 'none'
#--dsa_strategy_stage_2: 'none' 

```

### Visualization of accumulation

```
Mesh_vis.py 

```



### Performance comparison to other Data Augmentation and analysis of synergistic collaboration with other Data Augmentation
```
python main_creation_Tradition.py   
#--model: ConvNet 
#--dataset:CIFAR10
#--seedsetting: 1,2,3
#--ipc (images/class): 10
#--Iteration: 1000
#--dsa_strategy_stage_1: 'none'
#--dsa_strategy_stage_2: 'none'

Mesh_vis.py 

```