import torch
import torch.nn as nn

class Baseline(nn.Module):
    def __init__(self, sum_classes):
        super(Baseline, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.Flatten(),
            nn.Linear(12160, 32),
            nn.Linear(32,4),
            nn.Linear(4, sum_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    test = Baseline(17)
    input = torch.randn(64, 1, 190)
    output = test(input)
    print(output.shape)