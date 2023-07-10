from Net.UNet.UNet_Decoder import UNetDecode
from Net.UNet.UNet_Encoder import UNetEncode
import torch
import torch.nn as nn


class UNet(nn.Module):
     def __init__(self, output_channels, input_channels=3, p=0.5):
         super().__init__()
       
         self.encode_lvl_0 = UNetEncode(input_channels, 64, p)
         self.encode_lvl_1 = UNetEncode(64, 128, p)
         self.encode_lvl_2 = UNetEncode(128, 256, p)
         self.encode_lvl_3 = UNetEncode(256, 512, p)
         self.encode_lvl_4 = UNetEncode(512, 1024, p)
         self.decode_lvl_4 = UNetDecode(1024, 1024) 
         self.decode_lvl_3 = UNetDecode(512 + 1024, 512)
         self.decode_lvl_2 = UNetDecode(256 + 512, 256)
         self.decode_lvl_1 = UNetDecode(128 + 256, 128)
         self.decode_lvl_0 = UNetDecode(64 + 128, 64)
       
         self.output = nn.Conv2d(64, output_channels, 1)
       
     def forward(self, x):
         # Encode
         x = self.encode_lvl_0(x)
         x = self.encode_lvl_1(x)
         x = self.encode_lvl_2(x)
         x = self.encode_lvl_3(x)
         x = self.encode_lvl_4(x)
         # Decode
         x = self.decode_lvl_4(x)
         x = torch.cat([x, self.encode_lvl_3.getResult()], dim=1)
         x = self.decode_lvl_3(x)
         x = torch.cat([x, self.encode_lvl_2.getResult()], dim=1)
         x = self.decode_lvl_2(x)
         x = torch.cat([x, self.encode_lvl_1.getResult()], dim=1)
         x = self.decode_lvl_1(x)
         x = torch.cat([x, self.encode_lvl_0.getResult()], dim=1)
     
         x = self.decode_lvl_0(x)
         # Output
         return self.output(x)