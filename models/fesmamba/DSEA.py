import torch
from torch import nn


class Module3(nn.Module):
    def __init__(self, input_dim=64):
        super().__init__()

        self.input_dim = input_dim

        self.Sigmoid = nn.Sigmoid()

        self.conv = nn.Sequential(
            nn.Conv2d(input_dim * 2, input_dim, kernel_size=1), nn.BatchNorm2d(input_dim),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(input_dim),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(input_dim),
            nn.ReLU(True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=1), nn.BatchNorm2d(input_dim),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(input_dim),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(input_dim),
            nn.ReLU(True),
        )

        self.weight = nn.Sequential(
            nn.Conv2d(input_dim, input_dim // 16, 1, bias=True),
            nn.BatchNorm2d(input_dim // 16),
            nn.ReLU(True),
            nn.Conv2d(input_dim // 16, input_dim, 1, bias=True),
            nn.Sigmoid())

        self.norm = nn.BatchNorm2d(input_dim)
        self.relu = nn.ReLU(input_dim)

        self.out = nn.Sequential(
            nn.Conv2d(input_dim * 2, input_dim, kernel_size=3, padding=1), nn.BatchNorm2d(input_dim),nn.ReLU(True)
        )

    def forward(self, Hin_feature, Lin_feature):
        Los_feature = torch.cat([Hin_feature, Lin_feature], dim=1)
        Los_feature = self.conv(Los_feature)

        Hos_feature = torch.cat([Hin_feature, Lin_feature], dim=1)
        Hos_feature = self.conv(Hos_feature)

        L_feature = self.conv3(Los_feature)
        H_feature = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(Hos_feature.float()).real)*torch.fft.fft2(Hos_feature.float())))))

        g_L_feature = self.Sigmoid(L_feature)
        g_H_feature = self.Sigmoid(H_feature)

        L_feature = Lin_feature + Lin_feature * g_L_feature + (1 - g_L_feature) * g_H_feature * H_feature
        H_feature = Hin_feature + Hin_feature * g_H_feature + (1 - g_H_feature) * g_L_feature * L_feature

        out = self.out(torch.cat([H_feature, L_feature], dim=1))
        return out

if __name__ == '__main__':
    S = Module3(512).cuda()
    intput_1 = torch.rand(2, 512, 64, 64).to('cuda')
    input_2 = torch.rand(2, 512, 64, 64).to('cuda')
    out = S(intput_1, input_2)
    print(out.shape)