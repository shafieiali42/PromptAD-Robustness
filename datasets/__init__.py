import numpy as np
from torch.utils.data import DataLoader
from loguru import logger

from .dataset import CLIPDataset
from .mvtec import load_mvtec, mvtec_classes
from .visa import load_visa, visa_classes


mean_train = [0.48145466, 0.4578275, 0.40821073]
std_train = [0.26862954, 0.26130258, 0.27577711]

load_function_dict = {
    'mvtec': load_mvtec,
    'visa': load_visa,
}

dataset_classes = {
    'mvtec': mvtec_classes,
    'visa': visa_classes,
}

def denormalization(x):
    x = (((x.transpose(1, 2, 0) * std_train) + mean_train) * 255.).astype(np.uint8)
    return x

def get_dataloader_from_args(phase,corruption_func=None,severity_level=None, **kwargs):

    dataset_inst = CLIPDataset(
        load_function=load_function_dict[kwargs['dataset']],
        category=kwargs['class_name'],
        phase=phase,
        k_shot=kwargs['k_shot'],
        corruption_func=corruption_func,
        severity=severity_level
    )


    # from PIL import Image
    # import random
    # random_indx=random.randint(0,len(dataset_inst))
    # random_indx=0
    # image = Image.fromarray(dataset_inst[random_indx][0][:,:,::-1])
    # print(dataset_inst[random_indx][0][:,:,::-1].dtype)
    # image.save(f'test_result/image_{random_indx}.png')
    # corrupted = Image.fromarray(dataset_inst[random_indx][5][:,:,::-1])
    # print(dataset_inst[random_indx][5][::,::-1].dtype)
    # corrupted.save(f'test_result/corrupted_{random_indx}_severity{severity_level}.png')
    # image = Image.fromarray(dataset_inst[random_indx][1])
    # image.save(f'test_result/gt_{random_indx}_{dataset_inst[random_indx][4]}.png')
    # exit()

    if phase == 'train':
        data_loader = DataLoader(dataset_inst, batch_size=kwargs['batch_size'], shuffle=True,
                                  num_workers=0)
    else:
        data_loader = DataLoader(dataset_inst, batch_size=kwargs['batch_size'], shuffle=False,
                                 num_workers=0)


    # data = next(iter(data_loader))
    # print(data[0].shape)
    # from PIL import Image
    # random_indx=10
    # print("-"*80)
    # print(data[0][0].shape)
    
    # image = Image.fromarray(data[0].numpy()[random_indx,:,:,::-1])
    # # print(dataset_inst[random_indx][0][:,:,::-1].dtype)
    # image.save(f'test_result/dataloader_image_{random_indx}.png')
    # corrupted = Image.fromarray(data[5].numpy()[random_indx,:,:,::-1])
    # # print(dataset_inst[random_indx][5][::,::-1].dtype)
    # corrupted.save(f'test_result/data_loader_corrupted_{random_indx}_severity{severity_level}.png')
    # # image = Image.fromarray(data[1][random_indx,:,:])
    # # image.save(f'test_result/gt_{random_indx}_{data[random_indx][4]}.png')
    # print("Done")
    # exit()
    # debug_str = f"===> datasets: {kwargs['dataset']}, class name/len: {kwargs['class_name']}/{len(dataset_inst)}, batch size: {kwargs['batch_size']}"
    # # logger.info(debug_str)


    return data_loader, dataset_inst