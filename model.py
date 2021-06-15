import torch.nn as nn
from torch.cuda import amp

class FSRCNN_x(nn.Module):
    def __init__(self, scale_factor=3, num_channels=3, d=56, s=12, m=4):
        super(FSRCNN_x, self).__init__()
        self.extract_features = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU()
        )
        self.shrink = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU()
        )
        self.map = []
        for _ in range(m):
            self.map.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2)])
            self.map.extend([nn.PReLU()])
        self.map = nn.Sequential(*self.map)
        self.expand = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU()
        )
        self.deconv = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2, output_padding=scale_factor-1)
    
    def forward(self, x):
        x = self.extract_features(x)
        x = self.shrink(x)
        x = self.map(x)
        x = self.expand(x)
        x = self.deconv(x)
        return x