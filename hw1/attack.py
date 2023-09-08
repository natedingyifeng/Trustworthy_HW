import torch
import torch.nn as nn


# fix seed so that random initialization always performs the same 
torch.manual_seed(1)

is_cw = True


# create the model N as described in the question
N = nn.Sequential(nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 3, bias=False))
device = torch.device("cuda:0")
N.to(device)

epsReal = 0.15 #depending on your data this might be large or small
eps = epsReal - 1e-7 # small constant to offset floating-point erros

# random input
x = torch.rand((1,10)).to(device) # the first dimension is the batch size; the following dimensions the actual dimension of the data
x.requires_grad_() # this is required so we can compute the gradient w.r.t x

t = 0 # target class

# The network N classfies x as belonging to class 2
original_class = N(x).argmax(dim=1).item()  # TO LEARN: make sure you understand this expression
print("Original Class: ", original_class)
assert(original_class == 2)

# compute gradient
# note that CrossEntropyLoss() combines the cross-entropy loss and an implicit softmax function
L = nn.CrossEntropyLoss()
loss = L(N(x), torch.tensor([t], dtype=torch.long).to(device)) # TO LEARN: make sure you understand this line
loss.backward()

# your code here
# adv_x should be computed from x according to the fgsm-style perturbation such that the new class of xBar is the target class t above
# hint: you can compute the gradient of the loss w.r.t to x as x.grad
if is_cw:
    if t == 0:
        max_steps = 7
    elif t == 1:
        max_steps = 10
    c = 1
    w = torch.zeros((1,10)).to(device)
    w.requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr = 0.1)
    for current_step in range(max_steps):
        adv_x = x + w
        f_loss = L(N(adv_x), torch.tensor([t], dtype=torch.long).to(device)) - 1
        delta = torch.clamp(adv_x - x, min=-eps, max=eps)
        w_loss = torch.sum(delta + c*f_loss)
        optimizer.zero_grad()
        w_loss.backward()
        optimizer.step()
    new_class = N(adv_x).argmax(dim=1).item()
    print("New Class: ", new_class)
    assert(new_class == t)
    print(torch.norm((x-adv_x), p=float('inf')))
else:
    adv_x = x - eps * torch.sign(x.grad)
    new_class = N(adv_x).argmax(dim=1).item()
    print("New Class: ", new_class)
    assert(new_class == t)
    # it is not enough that adv_x is classified as t. We also need to make sure it is 'close' to the original x. 
    assert( torch.norm((x-adv_x), p=float('inf')) <= epsReal)
