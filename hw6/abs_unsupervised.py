# -*- coding: utf-8 -*-
"""Abs_Supervised.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aGirCJpCUl2lu5DKWFwW8X3blFpD9W6e

## Imports and Boilerplate
"""

from dl2 import dl2lib

import json
import matplotlib.pyplot as plt
import torch
import argparse

torch.manual_seed(42)

# Basic Model Class
class BasicModel(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(BasicModel, self).__init__()

        assert len(hidden_sizes) >= 1

        # Create hidden layers dynamically using a loop
        layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                # First hidden layer connected to the input
                layers.append(torch.nn.Linear(input_size, hidden_sizes[i]))
            else:
                # Subsequent hidden layers connected to the previous hidden layer
                layers.append(torch.nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))

            layers.append(torch.nn.ReLU())

        # Output layer
        layers.append(torch.nn.Linear(hidden_sizes[len(hidden_sizes) - 1], output_size))

        # Combine all layers into a sequential module
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, inp):
        return self.layers(inp)

# Model class that works on intervals
class IntervalNN(BasicModel):
    def __init__(self, num_interval_inputs, hidden_sizes, output_size):
        super().__init__(2*num_interval_inputs, hidden_sizes, output_size)

    def forward(self, *intervs):
        if len(intervs) == 1:
            inp = intervs[0]
        else:
            inp = torch.cat(tuple(intervs), dim = 1)

        return super().forward(inp)

"""## Loading the data"""

def load_data(file_name):
    with open(file_name, "r") as f:
        obj = json.load(f)

    intervs = []
    abs_ints = []

    for o in obj:
        intervs.append(torch.tensor(o["interv"]))
        abs_ints.append(torch.tensor(o["abs"]))


    intervs = torch.stack(intervs, dim = 0)
    abs_ints = torch.stack(abs_ints, dim = 0)

    return intervs, abs_ints

intervs, abs_ints = load_data("datasets/interv_abs_train_data.json")

is_supervised = False

parser = argparse.ArgumentParser()
parser = dl2lib.add_default_parser_args(parser)
args = parser.parse_args()

"""## Supervised training for sound learning of Interval Abs"""

def train_sound_abs_model(model, opt, epochs, intervs, abs_ints, is_supervised=True):
    loss_history = []

    for _ in range(epochs):
        opt.zero_grad()
        outp = model(intervs)

        # Write the logical constraint for soundness here using the DL library
        if is_supervised:
            constraints = [
                dl2lib.diffsat.LEQ(outp[:, 0], abs_ints[:, 0]),
                dl2lib.diffsat.GEQ(outp[:, 1], abs_ints[:, 1]),
            ] 
            soundness_loss = dl2lib.diffsat.And(constraints).loss(None).mean()
        else:
            constraints = [
                dl2lib.diffsat.LEQ(outp[:, 0], torch.zeros_like(outp[:, 0])),
                dl2lib.diffsat.GEQ(outp[:, 1], torch.maximum(intervs[:, 1], -intervs[:, 0])),
            ] 
            soundness_loss = dl2lib.diffsat.And(constraints).loss(None).mean()

        loss_history.append(soundness_loss.item())
        soundness_loss.backward()
        opt.step()

    return loss_history

sound_abs_model = IntervalNN(1,[8,16,8],2)
lr = 0.04
epochs = 500
opt = torch.optim.Adam(sound_abs_model.parameters(), lr)
loss_history = train_sound_abs_model(sound_abs_model, opt, epochs, intervs, abs_ints, is_supervised=is_supervised)
print(loss_history[-1])

plt.plot(loss_history)
plt.xlabel('optimization step')
plt.ylabel('loss')
if is_supervised:
    plt.savefig('supervised_sound_learning.png')
else:
    plt.savefig('unsupervised_sound_learning.png')
plt.show()

"""## Evaluate model's quality

After training our model, we check its quality by measuring 2 metrics:
1. Soundness: We use our model to find the absolute of the intervals in a test set and then check if the returned absolute interval is sound. As we have the ground truths in the test set, this is easy to check; the ground truth abs interval should be inside the predicted interval.
2. Imprecision: For the cases where the predicted answer is sound, we also measure how big the predicted interval is as compared to the ground truth.

Ideally, we want a model that is very sound and as precise it can be, i.e. one with **high soundness and low imprecision**.
"""

def evaluate_model(model):
    # load data from test file
    intervs, abs_ints = load_data("datasets/interv_abs_test_data.json")

    # get output from the model
    output = model(intervs)

    # Measuring soundness!
    ctr = 0
    unsound_ctr = 0

    # Difference in the ranges of intervals (only count in cases of sound answers)
    precision_sum = 0

    for into, intg in zip(output, abs_ints):
        ctr += 1

        if (into[0] > intg[0]) or (into[1] < intg[1]):
            unsound_ctr += 1
        else:
            # If sound, measure precision
            precision_sum += ((into[1]-into[0]) - (intg[1] - intg[0])).item()


    print("Soundness Measure:" , ((ctr-unsound_ctr)/ctr)*100, " ({} out of {})".format(ctr-unsound_ctr, ctr))

    if ctr != unsound_ctr:
        print("Imprecision Measure:" , (precision_sum/(ctr-unsound_ctr)), " ({} diff. in {} cases)".format(precision_sum, ctr-unsound_ctr))
    else:
        print("Imprecision Measure: -")

evaluate_model(sound_abs_model)

"""If the soundness loss is implemented correctly, then we will observe a very high soundness accuracy of the model. But observe that the imprecision measure is also very high. This is expected because in our loss method, we only optimized for soundness. This made the model learn very large intervals like say (0, 1000), when the ground truth abs is (1, 80). We will now handle this by also including a precision penalty in the loss method, which should incentivize the model to learn smaller intervals.

## Supervised training for sound *and precise* learning of Interval Abs

Add a precision loss to the training. The combined loss is a weighted combination of the soundness and precision losses. We set the weights 20:1 for now.
"""

def train_sound_and_prec_abs_model(model, opt, epochs, intervs, abs_intervs, soundness_weight, precision_weight, is_supervised=True):
    loss_history = []

    for _ in range(epochs):
        opt.zero_grad()
        outp = model(intervs)

        if is_supervised:
            # Write the same soundness loss as before here
            constraints_soundness = [
                dl2lib.diffsat.GEQ(outp[:, 0], torch.zeros_like(outp[:, 0])),
                dl2lib.diffsat.LEQ(outp[:, 0], abs_ints[:, 0]),
                dl2lib.diffsat.GEQ(outp[:, 1], abs_ints[:, 1]),
            ]
            soundness_loss = dl2lib.diffsat.And(constraints_soundness).loss(None).mean()
            # Write the precision loss here
            constraints_precision = [
                dl2lib.diffsat.EQ(outp[:, 0], abs_ints[:, 0]),
                dl2lib.diffsat.EQ(outp[:, 1], abs_ints[:, 1]),
            ]
            precision_loss = dl2lib.diffsat.And(constraints_precision).loss(None).mean()
        else:
            constraints = [
                dl2lib.diffsat.LEQ(outp[:, 0], torch.zeros_like(outp[:, 0])),
                dl2lib.diffsat.GEQ(outp[:, 1], torch.maximum(intervs[:, 1], -intervs[:, 0])),
            ] 
            soundness_loss = dl2lib.diffsat.And(constraints).loss(None).mean()
            constraints_precision = [
                dl2lib.diffsat.EQ(outp[:, 0], torch.maximum(torch.zeros_like(outp[:, 0]), -torch.minimum(intervs[:, 1], -intervs[:, 0]))),
                dl2lib.diffsat.EQ(outp[:, 1], torch.maximum(intervs[:, 1], -intervs[:, 0])),
            ]
            precision_loss = dl2lib.diffsat.And(constraints_precision).loss(None).mean()

        loss = soundness_weight * soundness_loss + precision_weight * precision_loss
        loss_history.append(loss.item())
        loss.backward()
        opt.step()

    return loss_history

sound_and_prec_abs_model = IntervalNN(1,[8,16,8],2)
lr = 0.04
epochs = 500
opt = torch.optim.Adam(sound_and_prec_abs_model.parameters(), lr)

"""## Studying Soundness vs Precision Tradeoff

Analyze the soundness/precision tradeoff by experimenting with different ratios, 20:1, 10:1, etc. Plot the results and report your observations.
"""
for (sw, pw) in [(20, 1), (10, 1), (5, 1), (1, 1), (1, 5), (1, 10), (1, 20)]:
    print('sw : pw = ' + str(sw) + ' : ' + str(pw))
    loss_history = train_sound_and_prec_abs_model(sound_and_prec_abs_model, opt, epochs, intervs, abs_ints, sw, pw, is_supervised=is_supervised)
    print(loss_history[-1])

    plt.clf()
    plt.plot(loss_history)
    plt.xlabel('optimization step')
    plt.ylabel('loss')
    if is_supervised:
        plt.savefig('supervised_sound_and_precise_learning_sw_' + str(sw) + '_pw_' + str(pw) + '.png')
    else:
        plt.savefig('unsupervised_sound_and_precise_learning_sw_' + str(sw) + '_pw_' + str(pw) + '.png')
    
    plt.show()

    evaluate_model(sound_and_prec_abs_model)

