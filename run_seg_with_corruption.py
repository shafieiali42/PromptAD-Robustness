import os
from datasets import dataset_classes
from multiprocessing import Pool

if __name__ == '__main__':

    pool = Pool(processes=1)


    datasets = ['mvtec', 'visa']
    shots = [1, 2, 4]
    corrupt_list=[False,True]
    corruptions=['shot_noise','gaussian_noise','impulse_noise']
    severities=[1,2,3]
    
    for shot in shots:
        for dataset in datasets:
            classes = dataset_classes[dataset]
            for cls in classes[:]:
                for corrupt in corrupt_list: 
                    if corrupt==False:
                        sh_method = f'python test_seg.py ' \
                                    f'--dataset {dataset} ' \
                                    f'--k-shot {shot} ' \
                                    f'--class_name {cls} ' \
                                    
                        print(sh_method)
                        pool.apply_async(os.system, (sh_method,))
                    else:
                        for corruption in  corruptions:
                            for severity in severities:
                                    sh_method = f'python test_seg.py ' \
                                                f'--dataset {dataset} ' \
                                                f'--k-shot {shot} ' \
                                                f'--class_name {cls} ' \
                                                f'--corrupt true '\
                                                f'--corruption {corruption} '\
                                                f'--severity {severity} '\

                                    print(sh_method)
                                    pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()

