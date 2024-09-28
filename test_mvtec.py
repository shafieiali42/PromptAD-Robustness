import cv2 as cv

import cv2 as cv
import numpy as np
   
from PIL import Image
import os
from utils.metrics import *




# from imagecorruptions import corruppythot



# def gaussian_noise(x, severity=1):
#     c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

#     x = np.array(x) / 255.
#     return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


# def shot_noise(x, severity=1):
#     c = [60, 25, 12, 5, 3][severity - 1]

#     x = np.array(x) / 255.
#     return np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255

import pandas as pd


def write_results(results:dict, cur_class, total_classes, csv_path):
    total_classes=total_classes+["mean"]
    keys_ = list(results.keys())
    type_of_metric=["i_roc","p_roc"]
    type_of_corruptions = ['no_curruption','shot_noise_1','shot_noise_2', 'shot_noise_3',
            'gaussian_noise_1', 'gaussian_noise_2','gaussian_noise_3',
            'impulse_noise_1','impulse_noise_2','impulse_noise_3']

    keys = [f"{a}_{b}" for a in type_of_metric for b in type_of_corruptions]
    print(f"Keys: {keys}")
    if not os.path.exists(csv_path):
        df_all = None
        for class_name in total_classes:
            r = dict()
            for k in keys:
                r[k] = 0.00
            df_temp = pd.DataFrame(r, index=[class_name])

            if df_all is None:
                df_all = df_temp
            else:
                df_all = pd.concat([df_all, df_temp], axis=0)

        df_all.to_csv(csv_path, header=True, float_format='%.2f')

    df = pd.read_csv(csv_path, index_col=0)

    for k in keys_:
        df.loc[cur_class, k] = results[k]

    df.to_csv(csv_path, header=True, float_format='%.2f')


def save_metric(metrics, total_classes, class_name,dataset,corrupt,corruption_type,severity, csv_path):
    # if dataset != 'mvtec':
    for indx in range(len(total_classes)):
        total_classes[indx] = f"{dataset}-{total_classes[indx]}"
    class_name = f"{dataset}-{class_name}"
    string_to_add=""
    if not corrupt:
        string_to_add="no_curruption"
    else:
        string_to_add=f"{corruption_type}_{severity}"

    print(f"Metric: {metrics}")
    new_metrics = {f"{k}_{string_to_add}": v for k, v in metrics.items()}
    metrics=new_metrics
    print(f"Metric: {metrics}")
    write_results(metrics, class_name, total_classes, csv_path)

metrics= {'i_roc': 93.54}

visa_classes = ['candle', 'capsules', 'cashew', 'chewinggum',
                   'fryum', 'macaroni1', 'macaroni2',
                'pcb1', 'pcb2', 'pcb3','pcb4', 'pipe_fryum']

save_metric(metrics, visa_classes, "cashew","visa",0, None,0,"csv_path.csv")
# image=cv.imread("mvtec_anomaly_detection/carpet/train/good/000.png")

# corrupted_image = corrupt(image, corruption_name='gaussian_blur', severity=1)
# cv.imwrite("corrupted_using_library.jpg",corrupted_image)

# print(image.shape)

# a=np.uint8(gaussian_noise(Image.fromarray(image),severity=5))
# print(a)
# cv.imwrite("corruped.jpg",a)

# tst_ldr = [(torch.stack([model.transform(Image.fromarray(f.numpy())) for f in bt[0]], 
#                         dim=0), bt[1], bt[2], bt[3],  bt[4]) for bt in dataloader] 
# There is no point in random shuffle on test loader ... better to be list if no augmentation