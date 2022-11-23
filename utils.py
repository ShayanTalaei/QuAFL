import torch
import matplotlib.pyplot as plt
import math

def put_state_dict_on_gpu(state_dict, gpu_id):
    new_state_dict = {}
    for key, param in state_dict.items():
        new_state_dict[key] = param.cuda(gpu_id)
    return new_state_dict
    
def plot_trends(trends, x_axis, y_axis, start = 0, end = float('inf'), save_in = None, dataset_folder = None, name = None):
    
    fig = plt.figure(figsize=(6, 4))
    fig.set_facecolor('white')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.grid(True)
    
    shapes = ["8", "s", "p", "P", "*", "h", "H", "x", "d", "D", "^", "<", ">", "x", "X", ]
    
    for i , case in enumerate(trends):
        trend = trends[case]
        X, Y = trend[x_axis], trend[y_axis]
        Z = zip(X, Y)
        X, Y = [], []
        for z in Z:
            if z[0] >= start and z[0] <= end:
                X.append(z[0])
                Y.append(z[1])
        count = len(X)
        plt.plot(X, Y, marker=shapes[i], label=case, markevery= math.ceil(count * 0.1))
    
    plt.legend()
    if name != None:
        plt.savefig(f'Results/{dataset_folder}/{name}_{y_axis}_{x_axis}.pdf') 
    plt.show() 
     