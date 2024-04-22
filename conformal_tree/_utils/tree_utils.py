from sklearn import tree
from sklearn import ensemble
import matplotlib.pyplot as plt
import numpy as np
import torch

def forest_membership(data, residuals, max_depth=20,
                    max_leaf_nodes=80,
                    min_samples_leaf=20,
                    n_estimators=100):

    rf_model = ensemble.RandomForestRegressor(
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        n_estimators=n_estimators
    )

    rf_model.fit(data, residuals)

    membership = rf_model.apply(data)
    return rf_model, membership


def tree_membership(data,c_gap,  max_depth=20, 
                    max_leaf_nodes=80,  min_samples_leaf=20):
    
    tree_model = tree.DecisionTreeRegressor(
                        max_depth=max_depth,
                        max_leaf_nodes=max_leaf_nodes,
                        min_samples_leaf=min_samples_leaf)
    
    tree_model.fit(data,c_gap)
    membership = tree_model.apply(data)
    return tree_model, membership

def tree_plotter(tree_model,x, c_gap, ax=None,title=""):
    
    x_grid = np.linspace(0, 1, 1000)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10,5))
        
    tree_out =tree_model.predict(x_grid.reshape(-1,1))
    ax.bar(x,c_gap, width=0.005)
    ax.fill_between(x_grid, 0*tree_out ,tree_out , color="blue", alpha=0.3)
    ax.set_title(title)
    
# Test: 
def tester(tree_model, test_loader, q_set, net, preprocess=lambda x: x.cpu()):

    soft= torch.nn.Softmax(dim=1)
    
    I_contains=[]
    C_sizes =[]
    Confuss=[]
    for i,(x,y) in enumerate(test_loader):
        
        membership = tree_model.apply(preprocess(x))
        mini_batch_q = torch.tensor([q_set[meb] for meb in membership]).float()

        hot_y = torch.eye(10)[y]

        with torch.no_grad():
            out = net(x).cpu()
            res= 1-(soft(out) *hot_y).sum(1)

        C_set_mini_batch = (soft(out)>1-mini_batch_q.view(-1,1))
        C_set_mini_batch = C_set_mini_batch | (soft(out)==(soft(out).max(1)[0].view(-1,1)))
        
        I_contain = (C_set_mini_batch*torch.eye(10)[y]).sum(1)
        C_size = C_set_mini_batch.sum(1)
        Confus = (hot_y.unsqueeze(2) * C_set_mini_batch.unsqueeze(1)).mean(0)

        I_contains.append(I_contain.numpy())
        C_sizes.append(C_size.numpy())
        Confuss.append(Confus.numpy())
    I_contains = np.concatenate(I_contains)
    C_sizes = np.concatenate(C_sizes)
    Confuss = np.stack(Confuss).mean(0)
    
    A_conf = I_contains.mean()
    E_T = C_sizes.mean()
    Co_F=(Confuss**2).sum()
    
    return A_conf,E_T,Co_F
