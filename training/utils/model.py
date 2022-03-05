import torch
from torch import nn


class LandmarkDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super(LandmarkDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p

    def generate_mask(self, landmark, frame):
        position_p = torch.bernoulli(torch.Tensor([1 - self.p]*(landmark//2)))
        return position_p.repeat(1, frame, 2)

    def forward(self, x: torch.Tensor):
        if self.training:
            _, frame, landmark = x.size()
            landmark_mask = self.generate_mask(landmark, frame)
            scale = 1/(1-self.p)
            return x*landmark_mask.to(x.device)*scale
        else:
            return x


class LRNet(nn.Module):
    def __init__(self, rnn_unit=64, dropout_rate=0.5):
        super(LRNet, self).__init__()

        self.dropout_landmark = LandmarkDropout(0.20)
        self.gru = nn.GRU(input_size=136, hidden_size=rnn_unit, batch_first=True, bidirectional=True)
        self.dropout_feature_1 = nn.Dropout(dropout_rate)
        self.linear_1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.dropout_feature_2 = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(64, 2)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dropout_landmark(x)
        x = self.gru(x)[0]
        x = x[:, -1, :]
        x = self.dropout_feature_1(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout_feature_2(x)
        x = self.linear_2(x)
        x = self.output(x)

        return x
