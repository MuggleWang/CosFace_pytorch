import torch
import torch.nn as nn


# ---------------------------------------------- new version ----------------------------------------------
# Simpler and more scalable
# You can change the network to sphere64 by changing the variables sphere.layers
class Block(nn.Module):
    def __init__(self, planes):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.prelu1 = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.prelu2 = nn.PReLU(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.prelu2(out)
        out += residual

        return out


class sphere(nn.Module):
    def __init__(self, is_gray=False):
        super(sphere, self).__init__()
        block = Block
        layers = [1, 2, 4, 1]
        filter_list = [3,64,128,256,512]
        if is_gray:
            filter_list[0] = 1

        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)
        self.fc = nn.Linear(512 * 7 * 6, 512)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0.0)

    def _make_layer(self, block, inplanes, planes, blocks, stride):
        layers = []
        layers.append(nn.Conv2d(inplanes, planes, 3, stride, 1))
        layers.append(nn.PReLU(planes))
        for i in range(blocks):
            layers.append(block(planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)

# ---------------------------------------------- old version ----------------------------------------------
class sphere20(nn.Module):
    def __init__(self):
        super(sphere20, self).__init__()

        # input = B*3*112*96
        # First layer of each block have bias (lr_mult: 2 && decay_mult: 0)
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)  # =>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.relu1_3 = nn.PReLU(64)

        # First layer of each block have bias (lr_mult: 2 && decay_mult: 0)
        self.conv2_1 = nn.Conv2d(64, 128, 3, 2, 1)  # =>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)  # =>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.relu2_5 = nn.PReLU(128)

        # First layer of each block have bias (lr_mult: 2 && decay_mult: 0)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 2, 1)  # =>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)  # =>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)  # =>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)  # =>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu3_9 = nn.PReLU(256)

        # First layer of each block have bias (lr_mult: 2 && decay_mult: 0)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 2, 1)  # =>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512 * 7 * 6, 512)

        # Weight initialization
        # print('Initialization Network Parameter...')
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.xavier_uniform(m.weight)
                    nn.init.constant(m.bias, 0.0)
                else:
                    m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

        x = x.view(x.size(0), -1)
        x = self.fc5(x)

        return x

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)
