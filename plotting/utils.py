import matplotlib.pyplot as plt
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
