import os
import matplotlib.pyplot as plt
from tbparse import SummaryReader
from pathlib import Path
# ann = 'le_net_cifar'
ann = 'resnet20'
parent_path = Path(r'D:\OneDrive\Documents\git_repo\biotorch\trained_models\cifar10') / ann

def set_mpl():
    import matplotlib as mpl
    mpl.rcParams['font.size'] = 8
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'arial'
    mpl.rcParams['savefig.dpi'] = 480

def plot_start(square=True,figsize=None,ticks_pos=True):
    '''
    unified plot params
    '''
    set_mpl()
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    elif square:
        fig = plt.figure(figsize=(1.5, 1.5))
    else:
        fig = plt.figure(figsize=(1.5, 0.8))
    ax = fig.add_axes((0.1,0.1,0.8,0.8))
    if ticks_pos:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    return fig,ax

def path2df(folder_path):
    # search for tf event files in folder
    for path in folder_path.rglob('*.tfevents.*'):
        path = str(path)
        print(path)
        reader = SummaryReader(path)
        df = reader.scalars
        return df
dfs = {}

if ann == 'le_net_cifar':
    layer_list = ['conv1_0', 'conv2_0', 'conv3_0','fc1_0', 'fc2_0']
elif ann == 'resnet20':
    layer_list = [f'conv{s // 10}_{s % 10}' for s in range(10, 29)] + ['fc_0']
else:
    raise NotImplementedError
for alg in ['backpropagation', 'fa', 'dfa', 'tfawd','usf','kpwd']:
    dfs[alg] = {}
    acc = path2df(parent_path / alg / 'logs')
    dfs[alg]['acc'] = acc[acc['tag'] == 'accuracy/test']
    if alg in ['fa', 'tfawd']:
        for metric_name in ['layer_alignment', 'weight_radio']:
            for layer in layer_list:
                metric = f'{metric_name}_train_{layer}'
                dfs[alg][metric] = path2df(parent_path / alg / 'logs' / metric)


fig, ax = plot_start(square=False)
for alg, alg_label in zip(['backpropagation', 'fa', 'dfa', 'tfawd','usf','kpwd'], ['BP', 'FA', 'DFA', 'PFA','SF','KP']):
    plt.plot(dfs[alg]['acc']['step'], dfs[alg]['acc']['value'], label=alg_label)
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