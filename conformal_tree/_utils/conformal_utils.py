from scipy.stats import norm
import matplotlib.pyplot as plt
import torch

def sorter(ref, tup):
    idxs = ref.argsort()
    new_tup=[]
    for data in tup:
        new_tup.append(data[idxs])
    return new_tup

def conf_plotter(x_cal,y_low,y_up, y_cal=None, test_data=None, 
                                   model=None,  ax=None, title=""):
    
    # test_data: tuple (x_test,y_test)
    # calib_data: tutple (x_cal, y_cal)
    # CIs: tuple (y_low,y_up)
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10,5))

    if model is not None: 
        x_grid = torch.linspace(0, 1, steps=1000).to(model.device).float().view(-1,1)
        with torch.no_grad():
            out_grid = model(x_grid)    
        ax.plot(x_grid.cpu(), out_grid.cpu())
    
    if test_data is not None:
        ax.scatter(*test_data, color="red", alpha=1, s=10, label='Test Samples')
    
    if y_cal is not None:
        ax.scatter(x_cal,y_cal, color="blue", alpha=1, s=10, label='Calib Samples')

    x_cal_sorted, y_low_sorted, y_up_sorted = sorter(x_cal, [x_cal, y_low,y_up])
    #ax.plot(x_cal_sorted, y_low_sorted, color="red")#, alpha=1, s=10, label='Test Samples')
    #ax.plot(x_cal_sorted, y_up_sorted, color="red")#, alpha=1, s=10, label='Test Samples')
    ax.fill_between(x_cal_sorted, y_low_sorted,y_up_sorted, color="blue", alpha=0.3)
    ax.set_title(title) 