import matplotlib.pyplot as plt
from cycler import cycler

def plotter(y_train, y_val, path, resolution, number_of_n2, number_of_puiss):
    
    # for dark theme
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = "#00000080"
    plt.rcParams['axes.facecolor'] = "#00000080"
    plt.rcParams['savefig.facecolor'] = "#00000080"
    # plt.rcParams['savefig.transparent'] = True
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Liberation Sans']
    # for plots
    tab_colors = ['tab:blue', 'tab:orange', 'forestgreen', 'tab:red', 'tab:purple', 'tab:brown',
                'tab:pink', 'tab:gray', 'tab:olive', 'teal']
    fills = ['lightsteelblue', 'navajowhite', 'darkseagreen', 'lightcoral', 'violet', 'indianred',
            'lavenderblush', 'lightgray', 'darkkhaki', 'darkturquoise']
    edges = tab_colors
    custom_cycler = (cycler(color=tab_colors)) + \
        (cycler(markeredgecolor=edges))+(cycler(markerfacecolor=fills))
    plt.rc('axes', prop_cycle=custom_cycler)
    plt.plot(y_train, label="Training Loss")
    plt.plot(y_val, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Losses")
    plt.legend()
    plt.savefig(f"{path}/losses_w{resolution}_n2{number_of_n2}_puiss{number_of_puiss}_2D.png")
    plt.close()
    