import torch
import torch.nn as nn

class Module2(nn.Module):
    def __init__(self, channels, in_channels):
        super(Module2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, in_channels, 1), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )

        self.Dconv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 3, padding=3,dilation=3), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )

        self.Dconv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 3, padding=5,dilation=5), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )
        self.Dconv7 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 3, padding=7,dilation=7), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )
        self.Dconv9 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 3, padding=9,dilation=9), nn.BatchNorm2d(in_channels),nn.ReLU(True)
        )

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels * 5, in_channels, 1), nn.BatchNorm2d(in_channels),nn.ReLU(True)
        )

        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1), nn.BatchNorm2d(in_channels//2), nn.ReLU(True),
            nn.Conv2d(in_channels//2, in_channels, kernel_size=1)
        )

        self.weight = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1, bias=True),
            nn.BatchNorm2d(in_channels // 16),
            nn.ReLU(True),
            nn.Conv2d(in_channels // 16, in_channels, 1, bias=True),
            nn.Sigmoid())

        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(True)

        self.average_channel_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_channel_pooling = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

        self.Sigmoid = nn.Sigmoid()

        self.out2 = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=3, padding=1), nn.BatchNorm2d(in_channels)
        )

    def forward(self, F1):
       F1_input  = self.conv1(F1)

       F1_3_t = self.Dconv3(F1_input)
       F1_3_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(F1_3_t.float()).real)*torch.fft.fft2(F1_3_t.float())))))
       F1_3 = torch.add(F1_3_t,F1_3_f)
       F2_3_avg = self.average_channel_pooling(F1_3)
       F2_3_max = self.max_channel_pooling(F1_3)
       F2_3_min = - self.max_channel_pooling(-F1_3)

       F1_5_t = self.Dconv5(F1_input + F1_3)
       F1_5_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(F1_5_t.float()).real)*torch.fft.fft2(F1_5_t.float())))))
       F1_5 = torch.add(F1_5_t, F1_5_f)
       F2_5_avg = self.average_channel_pooling(F1_5)
       F2_5_max = self.max_channel_pooling(F1_5)
       F2_5_min = - self.max_channel_pooling(-F1_5)

       F1_7_t = self.Dconv7(F1_input + F1_5)
       F1_7_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(F1_7_t.float()).real)*torch.fft.fft2(F1_7_t.float())))))
       F1_7 = torch.add(F1_7_t, F1_7_f)
       F2_7_avg = self.average_channel_pooling(F1_7)
       F2_7_max = self.max_channel_pooling(F1_7)
       F2_7_min = - self.max_channel_pooling(-F1_7)

       F1_9_t = self.Dconv9(F1_input + F1_7)
       F1_9_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(F1_9_t.float()).real)*torch.fft.fft2(F1_9_t.float())))))
       F1_9 = torch.add(F1_9_t, F1_9_f)
       F2_9_avg = self.average_channel_pooling(F1_9)
       F2_9_max = self.max_channel_pooling(F1_9)
       F2_9_min = - self.max_channel_pooling(-F1_9)

       F2_avg = (F2_3_avg + F2_5_avg + F2_7_avg + F2_9_avg)
       F2_avg = self.fc(F2_avg)

       F2_max = (F2_3_max + F2_5_max + F2_7_max + F2_9_max)
       F2_max = self.fc(F2_max)

       F2_min = (F2_3_min + F2_5_min + F2_7_min + F2_9_min)
       F2_min = self.fc(F2_min)

       F2_final = self.out2(torch.cat([F2_avg,F2_max,F2_min], dim=1))
       F2_final = self.Sigmoid(F2_final)

       F3_3 = F1_3 * F2_final
       F3_5 = F1_5 * F2_final
       F3_7 = F1_7 * F2_final
       F3_9 = F1_9 * F2_final

       F_out = self.out(self.reduce(torch.cat((F3_3,F3_5,F3_7,F3_9,F1_input),1)) + F1_input )

       return F_out

if __name__ == '__main__':
    Module2 = Module2(512,128).cuda()
    intput=torch.rand(2,512,64,64).to('cuda')
    out=Module2(intput)
    print(out.shape)
