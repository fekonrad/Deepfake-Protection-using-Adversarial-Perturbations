import numpy as np 
import torch
from torch import nn


def fast_gradient_sign(model, img, label, eta=0.1, max_steps=1):
    """
        Generates an adversarial perturbation of image `img` such the a classifier `model`
        classifies the perturbed image as `label`.
        :param model: nn.Module representing the classifier
        :param img: torch.tensor of shape (b, c, h, w); number of channels `c` should be compatible with model forward
        :param label: torch.tensor of shape (b, num_classes) representing the one-hot encoding of the false label
        :param eta: float representing the step-size in the procedure
        :max_steps: int representing the largest amount of steps taken by the method. Iteration terminates if
                    `max_steps` is reached or if model misclassifies the image as `label`
        :returns: torch.tensor of the same shape as `img` representing the adversarial perturbation added to the image.
    """
    loss_fn = nn.CrossEntropyLoss()
    delta = torch.zeros(img.shape, requires_grad=True)

    model.eval()
    for param in model.parameters():            # freeze all trainable parameters
        param.requires_grad = False

    pred = model(img + delta)
    pred_int = torch.argmax(pred).item()
    label_int = torch.argmax(label).item()
    steps = 0
    print(f"Base Prediction Logits: \n {pred.detach().numpy()[0]}")
    while np.abs(pred_int - label_int) > 0.1 and steps < max_steps:
        loss = loss_fn(pred, label)
        loss.backward()

        with torch.no_grad():
            delta -= eta * torch.sign(delta.grad)
        steps += 1
        pred = model(img + delta)
        pred_int = torch.argmax(pred).item()
        print(f"Number of steps done: {steps} \n Current Prediction Logits: \n {pred.detach().numpy()[0]}")

    return delta
