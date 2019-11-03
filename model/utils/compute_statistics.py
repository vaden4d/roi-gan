import torch
from tqdm import tqdm

def compute_mean_std(dataset):

    mean_acc = torch.zeros(3)
    variance_acc = torch.zeros(3)
    
    with tqdm(ascii=True, leave=False,
            total=len(dataset)) as bar:
        
        for i in range(len(dataset)):
            current = dataset[i]
            current_mean = current.unsqueeze(0).float().mean(axis=[0, 2, 3])
            current_sq_mean = (current**2).unsqueeze(0).float().mean(axis=[0, 2, 3])
            
            mean_acc += (current_mean - mean_acc) / (i+1)
            variance_acc += (current_sq_mean - variance_acc) / (i+1) 
            bar.update()
            
    variance_acc -= mean_acc**2
            
    std = torch.sqrt(variance_acc)

    return mean_acc, std