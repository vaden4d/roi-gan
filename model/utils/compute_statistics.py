import torch
from tqdm import tqdm

def compute_mean_std(dataset):

    mean_acc = torch.zeros(3)
    variance_acc = torch.zeros(3)
    
    with tqdm(ascii=True, leave=False,
            total=len(dataset)) as bar:
        
        for i in range(len(dataset)):
            current = dataset[i].unsqueeze(0).float().mean(axis=[0, 2, 3])
            prev_mean_acc = mean_acc
            mean_acc += (current - mean_acc) / (i+1)
            variance_acc += (current - prev_mean_acc) * (current - mean_acc)
            std = torch.sqrt(variance_acc / (i+1))

            bar.update()

    return mean_acc, std