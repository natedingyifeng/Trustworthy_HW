from pathlib import Path
from robustbench.data import load_cifar10c, load_cifar10
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import torchvision
import numpy as np


ROOT_DATASET_PATH = "/coconut/yifeng-data/surgical-finetuning/Datasets"

def get_loaders(cfg, corruption_type, severity):
    if cfg.data.dataset_name == "cifar10":
        x_corr, y_corr = load_cifar10c(
            10000, severity, ROOT_DATASET_PATH, False, [corruption_type]
        )
        x_clean, y_clean = load_cifar10(n_examples=10000)
        assert cfg.args.train_n <= 9000
        labels = {}
        num_classes = int(max(y_corr)) + 1
        for i in range(num_classes):
            labels[i] = [ind for ind, n in enumerate(y_corr) if n == i]
        num_ex = cfg.args.train_n // num_classes
        tr_idxs = []
        val_idxs = []
        test_idxs = []
        for i in range(len(labels.keys())):
            np.random.shuffle(labels[i])
            tr_idxs.append(labels[i][:num_ex])
            val_idxs.append(labels[i][num_ex:num_ex+10])
            test_idxs.append(labels[i][num_ex+10:num_ex+100])
        tr_idxs = np.concatenate(tr_idxs)
        val_idxs = np.concatenate(val_idxs)
        test_idxs = np.concatenate(test_idxs)
        
        tr_dataset = TensorDataset(x_corr[tr_idxs], y_corr[tr_idxs])
        val_dataset = TensorDataset(x_corr[val_idxs], y_corr[val_idxs])
        te_dataset = TensorDataset(x_corr[test_idxs], y_corr[test_idxs])

        tr_clean_dataset = TensorDataset(x_clean[tr_idxs], y_clean[tr_idxs])
    
    elif cfg.data.dataset_name == "imagenet-c":
        data_root = Path(ROOT_DATASET_PATH)
        image_dir = data_root / "ImageNet-C" / corruption_type / str(severity)
        dataset = ImageFolder(image_dir, transform=transforms.ToTensor())
        indices = list(range(len(dataset.imgs))) #50k examples --> 50 per class
        assert cfg.args.train_n <= 20000
        image_dir_clean = data_root / "ImageNet" / "val"
        dataset_clean = ImageFolder(image_dir_clean, transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]))
        labels = {}
        y_corr = dataset.targets
        y_clean = dataset_clean.targets
        for i in range(max(y_corr)+1):
            labels[i] = [ind for ind, n in enumerate(y_corr) if n == i] 
        num_ex = cfg.args.train_n // (max(y_corr)+1)
        tr_idxs = []
        val_idxs = []
        test_idxs = []
        for i in range(len(labels.keys())):
            np.random.shuffle(labels[i])
            tr_idxs.append(labels[i][:num_ex])
            val_idxs.append(labels[i][num_ex:num_ex+10])
            test_idxs.append(labels[i][num_ex+10:num_ex+20])
        tr_idxs = np.concatenate(tr_idxs)
        val_idxs = np.concatenate(val_idxs)
        test_idxs = np.concatenate(test_idxs)
        tr_dataset = Subset(dataset, tr_idxs)
        val_dataset = Subset(dataset, val_idxs)
        te_dataset = Subset(dataset, test_idxs)
        tr_clean_dataset = Subset(dataset_clean, tr_idxs)

    loaders = {}	

    trainloader = DataLoader(
        tr_dataset,
        batch_size=cfg.data.batch_size*cfg.data.gpu_per_node,
        num_workers=cfg.data.num_workers,
    )

    valloader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size*cfg.data.gpu_per_node, 
        num_workers=cfg.data.num_workers,
    )  

    testloader = DataLoader(
        te_dataset,
        batch_size=cfg.data.batch_size*cfg.data.gpu_per_node, 
        num_workers=cfg.data.num_workers,
    )

    traincleanloader = DataLoader(
        tr_clean_dataset,
        batch_size=cfg.data.batch_size*cfg.data.gpu_per_node,
        num_workers=cfg.data.num_workers,
    )

    loaders["train"] = trainloader
    loaders["val"] = valloader
    loaders["test"] = testloader
    loaders["train_clean"] = traincleanloader
    
    return loaders
