import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, activation_function):
        super().__init__()
        
        self.C = 2
        # Layer1: firstconv
        self.deepconv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 5), stride=(1, 1), padding=(0, 25), bias=False),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(2, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(num_features=25, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            activation_function,
            nn.MaxPool2d((1, 2)),
            nn.Dropout(p=0.5),

            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1, 5), stride=(1, 1), groups=25, padding=(0, 2), bias=False),
            nn.BatchNorm2d(num_features=50, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            activation_function,
            nn.MaxPool2d((1, 2)),
            nn.Dropout(p=0.5),

            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1, 5), stride=(1, 1), groups=50, padding=(0, 2), bias=False),
            nn.BatchNorm2d(num_features=100, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            activation_function,
            nn.MaxPool2d((1, 2)),
            nn.Dropout(p=0.5),

            nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(1, 5), stride=(1, 1), groups=100, padding=(0, 2), bias=False),
            nn.BatchNorm2d(num_features=200, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            activation_function,
            nn.MaxPool2d((1, 2)),
            nn.Dropout(p=0.5)
        )

        #self.convblock()

        # Layer4: classify
        self.classify = nn.Sequential(
            nn.Linear(in_features=9800, out_features=2, bias=True),
            nn.Softmax()
        )

    def forward(self, x):
        #x = x.float()
        
        x = self.deepconv(x)
        #print(x.shape[1]*x.shape[3])
        #a = input("eee")
        x = x.view(x.shape[0], 9800)
        x = self.classify(x)
        
        return x