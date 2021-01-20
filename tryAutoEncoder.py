import torch
from sequitur.models import CONV_LSTM_AE
from sequitur import quick_train

train_set = [torch.randn(10, 5, 5) for _ in range(100)]
encoder, decoder, _, _ = quick_train(CONV_LSTM_AE, train_set, encoding_dim=4)