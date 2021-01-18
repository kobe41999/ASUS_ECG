import torch.nn as nn
import torch.nn.functional as F
import torch


class RNN(nn.Module):
    def __init__(self, num_hiddens, num_layers, bidirectional, labels):
        super(RNN, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.encoder = nn.LSTM(input_size=1, hidden_size=self.num_hiddens,
                               num_layers=num_layers, bidirectional=self.bidirectional,
                               dropout=0)
        if self.bidirectional:
            self.decoder = nn.Sequential(
                # nn.Linear(num_hiddens * 10, 50),
                # nn.Linear(num_hiddens * 6, 64),
                # nn.Linear(num_hiddens * 10, 128),
                nn.Linear(num_hiddens * 8, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, labels),
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(num_hiddens * 4, 512),
                nn.ReLU(),
                nn.Linear(512, labels),
            )
        self.decoderForOneState = nn.Sequential(
            nn.Linear(num_hiddens * 2, labels),
        )

    def forward(self, inputs):
        # unpadInput=torch.nn.utils.rnn.pack_padded_sequence(inputs,sorted_length,batch_first=True)
        states, hidden = self.encoder(inputs)
        # states, unpacked_len=torch.nn.utils.rnn.pad_packed_sequence(states,batch_first=True,padding_value=-5)
        states = states.permute(0, 2, 1)  # 8 250 2700
        # print(states.shape)  #32 1024 65
        # pooling 是可以降低維度的方式
        avg_pool = F.adaptive_avg_pool1d(states, 2).contiguous().view(states.size(0), -1)  # 8 250 1->8 250
        max_pool = F.adaptive_max_pool1d(states, 2).contiguous().view(states.size(0), -1)

        out = torch.cat([avg_pool, max_pool], dim=1)
        outputs = self.decoder(out)
        return outputs

