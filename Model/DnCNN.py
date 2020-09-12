from torch import nn

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x)
        return x

# reference: original DnCNN
class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers): #num_of_layers - 2
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(in_channels=features, out_channels=3, kernel_size=kernel_size, padding=padding, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out

# DnCNN as the transformation layer
class DnCNN_I(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN_I, self).__init__()
        kernel_size = 3
        padding = 1
        features = channels # 64
        layers = []

        for _ in range(num_of_layers):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.InstanceNorm2d(features))
            layers.append(nn.Tanh())

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out

class DnCNN_II(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN_II, self).__init__()
        kernel_size = 3
        padding = 1
        features = channels # 64
        layers = []

        for _ in range(num_of_layers):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.InstanceNorm2d(features))
            layers.append(nn.Tanh())

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out+x

# class ResUnit(nn.Module):
#     def __init__(self, channels):
#         super(ResUnit, self).__init__()
#         kernel_size = 3
#         padding = 1
#         features = channels # 64
#         self.block1 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
#                           bias=False)
#         self.block2 = nn.InstanceNorm2d(features)
#         self.block3 = nn.Tanh()
#
#     def forward(self, x):
#         out = self.block1(x)
#         out = self.block2(out)
#         out = self.block3(out)
#         return x+out
#
# class DnCNN_II(nn.Module):
#     def __init__(self, channels, num_of_layers=17):
#         super(DnCNN_II, self).__init__()
#         features = channels # 64
#         layers = []
#         for _ in range(num_of_layers): #num_of_layers - 2
#             layers.append(ResUnit(features))
#
#         self.dncnn = nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.dncnn(x)
#         return out+x

