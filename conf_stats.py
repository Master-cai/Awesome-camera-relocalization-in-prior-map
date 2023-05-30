import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_one_conf(file_name, if_save=True, if_vis=False):
    print(f'plotting {file_name}...')
    with open(f'json_data/{file_name}', 'r') as f:
            data = json.load(f)
    # load all conf info
    
    # confs = {
    #   'CVPR': {
    #       '2017': 7,
    # }
    # conut by group
    confs = {}
    for d in data:
        # first see conf, add it if not exist
        if d['pub'] not in confs.keys():
            conf = {}
            # conf['name'] = d['pub']
            confs[d['pub']] = conf
        # conf exist, see year
        if d['year'] not in confs[d['pub']].keys():
            confs[d['pub']][d['year']] = 1
        else:
            confs[d['pub']][d['year']] += 1
    # plot confs by Histogram grouped by d['pub']
    sns.set_theme(style="whitegrid")

    total_num = 0
    # conf is less than 4, plot in one row 

    fig, axes = plt.subplots(len(confs.keys())//4+1, 4, figsize=(4*4, (len(confs.keys())//4+1)*4), squeeze=False)

    for i, (k, v) in enumerate(confs.items()):
        sns.barplot(x=list(v.keys()), y=list(v.values()), ax=axes[i//4, i%4])
        # show total number in title
        axes[i//4, i%4].set_title(f'{k} ({sum(v.values())})')
        total_num += sum(v.values())
        axes[i//4, i%4].set_xlabel('year')
        axes[i//4, i%4].set_ylabel('count')
        # set ytick interval to 1
        axes[i//4, i%4].set_yticks(np.arange(0, max(v.values())+1, 1))

    # add title and show total number
    fig.suptitle(f'{file_name.split(".")[0]} ({total_num})')
    # fig.suptitle(file_name.split(".")[0])

    plt.tight_layout()
    if if_vis:
        
        plt.show()
    if if_save:
    # save to pdf
        plt.savefig(f'conf_stats/{file_name.split(".")[0]}.pdf', bbox_inches='tight')


if __name__ == "__main__":
    # if len(sys.argv) == 2:
    #     file_name = sys.argv[1]
    # else:
    #     file_name = "Image appearance normalization.json"
    
    # plot_one_conf(file_name, if_save=True, if_vis=False)

    # plot all conf in 'json_data' folder
    for file_name in os.listdir('json_data'):
        plot_one_conf(file_name, if_save=True, if_vis=False)
