import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, activation_function):
        super().__init__()
        
        # Layer1: firstconv
        self.firstconv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(num_features=16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )

        # Layer2: depthwiseConv
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(num_features=32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            activation_function,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25)
        )
        
        # Layer3: separableConv
        self.separableConv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(num_features=32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            activation_function,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25)
        )
        
        # Layer4: classify
        self.classify = nn.Sequential(nn.Linear(in_features=736, out_features=2, bias=True))

    def forward(self, x):
        #x = x.float()
        #print(x.shape)
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(x.shape[0], 736) # x.shape[0] is batch size, may vary if run out of data
        x = self.classify(x)
        return x