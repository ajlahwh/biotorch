import os
import matplotlib.pyplot as plt
from tbparse import SummaryReader
from pathlib import Path
from utils import *

parent_path = Path(r'D:\OneDrive\Documents\git_repo\biotorch\trained_models')

## cifar10
# # ann = 'le_net_cifar'
# # layer_list = ['conv1_0', 'conv2_0', 'conv3_0','fc1_0', 'fc2_0']
# ann = 'resnet20'
# layer_list = [f'conv{s // 10}_{s % 10}' for s in range(10, 29)] + ['fc_0']
# parent_path = parent_path/ 'cifar10' / ann

## mnist
ann = 'ffn2'
layer_list = ['fc1', 'fc2', 'fc']
parent_path = parent_path/ 'mnist' / ann

def path2df(folder_path):
    # search for tf event files in folder
    for path in folder_path.rglob('*.tfevents.*'):
        path = str(path)
        print(path)
        reader = SummaryReader(path)
        df = reader.scalars
        return df
    return None
dfs = {}

color_map = {'backpropagation': 'C0', 'fa': 'C1', 'dfa': 'C2', 'tfawd': 'C3', 'usf': 'C4', 'kpwd': 'C5'}

for alg in ['backpropagation', 'fa', 'dfa', 'tfawd','usf','kpwd']:
    acc = path2df(parent_path / alg / 'logs')
    if acc is None:
        continue
    dfs[alg] = {}
    dfs[alg]['acc'] = acc[acc['tag'] == 'accuracy/test']
    if alg in ['fa', 'tfawd']:
        for metric_name in ['layer_alignment', 'weight_radio']:
            for layer in layer_list:
                metric = f'{metric_name}_train_{layer}'
                dfs[alg][metric] = path2df(parent_path / alg / 'logs' / metric)


fig, ax = plot_start(square=False)
for alg, alg_label in zip(['backpropagation', 'fa', 'dfa', 'tfawd','usf','kpwd'], ['BP', 'FA', 'DFA', 'PFA','SF','KP']):
    if alg not in dfs:
        continue
    plt.plot(dfs[alg]['acc']['step'], dfs[alg]['acc']['value'], label=alg_label, color=color_map[alg])
plt.xticks([0, 100, 200])
plt.yticks([0, 20, 40, 60, 80, 100])
plt.xlim(0, 200)
plt.ylim(0, 100)
plt.xlabel('Epoch')
plt.ylabel('Test accuracy')
plt.legend()
os.makedirs(Path('./figures') / ann, exist_ok=True)
plt.savefig(Path('./figures') /ann / 'test_acc.pdf', bbox_inches='tight')
plt.show()

for alg in ['fa', 'tfawd']:
    if alg not in dfs:
        continue
    for metric_name in ['layer_alignment', 'weight_radio']:
        fig, ax = plot_start(square=False)
        for layer_idx, layer in enumerate(layer_list):
            metric = f'{metric_name}_train_{layer}'
            plt.plot(dfs[alg][metric]['step'], dfs[alg][metric]['value'], color='k', alpha=0.5, linewidth=0.5
                     #label=layer[:-2], color=f'C{layer_idx+4}'
                     )
        plt.xticks([0, 100, 200])
        # plt.yticks([0, 0.5, 1])
        plt.xlim(0, 200)
        plt.ylim({'layer_alignment': (0, 100), 'weight_radio': (0, 3)}[metric_name])
        plt.yticks({'layer_alignment': [0, 50, 100], 'weight_radio': [0.5, 1, 2]}[metric_name])
        # plt.ylim(0, 1)
        # plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel({'layer_alignment': 'Weight alignment ($^\circ$)', 'weight_radio': 'Weight ratio'}[metric_name])
        os.makedirs(Path('./figures'), exist_ok=True)
        plt.savefig(Path('./figures')  /ann/ f'{alg}_{metric_name}.pdf', bbox_inches='tight')
        plt.show()