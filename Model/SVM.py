import torch.nn as nn


class SVM(nn.Module):
    """
    Linear Support Vector Machine
    -----------------------------
    This SVM is a subclass of the PyTorch nn module that
    implements the Linear  function. The  size  of  each
    input sample is 2 and output sample  is 1.
    """

    def __init__(self):
        super().__init__()  # Call the init function of nn.Module
        self.fully_connected = nn.Linear(2, 1)  # Implement the Linear function

    def forward(self, x):
        fwd = self.fully_connected(x)  # Forward pass
        return fwd
