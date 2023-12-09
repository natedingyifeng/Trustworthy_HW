import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, datasets, transforms
import numpy as np
import random
import time
from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, TensorDataset

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 64

np.random.seed(42)
torch.manual_seed(42)


def pgd_untargeted(model, x, labels, k, eps, eps_step):
    adv_x = x.clone().detach().requires_grad_()
    L = nn.CrossEntropyLoss()
    for i in range(k):
        adv_x_ = adv_x.clone().detach().requires_grad_()
        loss = L(model(adv_x_), labels)
        loss.backward()
        adv_x = adv_x + eps_step * torch.sign(adv_x_.grad)
        adv_x = torch.clamp(adv_x, x - eps, x + eps)
        adv_x = torch.clamp(adv_x, 0, 1)
    return adv_x


def adversarial_training(model, num_epochs):
    learning_rate = 1e-4

    opt = optim.Adam(params=model.parameters(), lr=learning_rate)

    ce_loss = torch.nn.CrossEntropyLoss()

    tot_steps = 0

    for epoch in range(1,num_epochs+1):
        t1 = time.time()
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            model.eval()
            x_batch = pgd_untargeted(model, x_batch, y_batch, k, eps, eps_step)
            model.train()
            tot_steps += 1
            opt.zero_grad()
            out = model(x_batch)
            batch_loss = ce_loss(out, y_batch)
            batch_loss.backward()
            opt.step()

        tot_test, tot_acc = 0.0, 0.0
        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            out = model(x_batch)
            pred = torch.max(out, dim=1)[1]
            acc = pred.eq(y_batch).sum().item()
            tot_acc += acc
            tot_test += x_batch.size()[0]
        t2 = time.time()

        print('Epoch %d: Accuracy %.5lf [%.2lf seconds]' % (epoch, tot_acc/tot_test, t2-t1))


def get_loaders():
    data_root = Path("/coconut/yifeng-data/surgical-finetuning/Datasets")
    image_dir_clean = data_root / "ImageNet" / "val"
    dataset_clean = ImageFolder(image_dir_clean, transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]))
    num_train = 1000
    indices = random.sample(list(range(len(dataset_clean))), num_train)
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * num_train))
    train_idx, test_idx = indices[split:], indices[:split]

    tr_dataset = Subset(dataset_clean, train_idx)
    te_dataset = Subset(dataset_clean, test_idx)

    trainloader = DataLoader(
        tr_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    testloader = DataLoader(
        te_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    return trainloader, testloader



if __name__ == '__main__':
    ## Dataloaders
    train_loader, test_loader = get_loaders()

    model = models.resnet18(pretrained=True)
    model = model.to(device)

    # test original accuracy
    model.eval()
    tot_test, tot_acc = 0.0, 0.0
    t1 = time.time()
    for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        out = model(x_batch)
        pred = torch.max(out, dim=1)[1]
        acc = pred.eq(y_batch).sum().item()
        tot_acc += acc
        tot_test += x_batch.size()[0]
    t2 = time.time()
    print('Original Accuracy: %.5lf [%.2lf seconds]' % (tot_acc/tot_test, t2-t1))

    k = 50
    for eps in [0.001, 0.01, 0.1]:
        print("epsilon:", eps)
        eps_step = 1e-4
        num_epochs = 5

        # Test PGD attack on original model
        tot_test, tot_acc = 0.0, 0.0
        t1 = time.time()
        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_batch = pgd_untargeted(model, x_batch, y_batch, k, eps, eps_step)
            out = model(x_batch)
            pred = torch.max(out, dim=1)[1]
            acc = pred.eq(y_batch).sum().item()
            tot_acc += acc
            tot_test += x_batch.size()[0]
        t2 = time.time()
        print('PGD on original model: Accuracy %.5lf [%.2lf seconds]' % (tot_acc/tot_test, t2-t1))

        # start adversarial training
        model.train()
        print("PGD adversarial training:")
        adversarial_training(model, num_epochs)
        model.eval()
        # Test PGD attack on enhanced model
        tot_test, tot_acc = 0.0, 0.0
        t1 = time.time()
        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_batch = pgd_untargeted(model, x_batch, y_batch, k, eps, eps_step)
            out = model(x_batch)
            pred = torch.max(out, dim=1)[1]
            acc = pred.eq(y_batch).sum().item()
            tot_acc += acc
            tot_test += x_batch.size()[0]
        t2 = time.time()
        print('PGD on enhanced model: Accuracy %.5lf [%.2lf seconds]' % (tot_acc/tot_test, t2-t1))
        torch.save(model, "models/resnet18_pgd_enhanced_eps_" + str(eps) + ".pt")