import pandas as pd
import os


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
    write_results(metrics, class_name, total_classes, csv_path)
