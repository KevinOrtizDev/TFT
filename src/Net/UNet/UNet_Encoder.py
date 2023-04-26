
import torch.nn as nn

class UNetEncode(nn.Module):
     def __init__(self, in_channels, out_channels, p=.5):
         super().__init__()
         self.encode = nn.Sequential(
             nn.Conv2d(in_channels, out_channels, 3, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(out_channels, out_channels, 3, padding=1),
             nn.Dropout2d(p),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(2, stride=2, ceil_mode=True)
         )

         self._result = None

     def forward(self, x):
         self._result = self.encode(x)
         return self._result

     def getResult(self):
         return self._result    