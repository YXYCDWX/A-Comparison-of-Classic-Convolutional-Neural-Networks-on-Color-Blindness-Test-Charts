import torch
import torch.nn as nn


class InceptionV1(nn.Module):
    def __init__(self, num_classes=10):
        super(InceptionV1, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.inception3a = self._inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = self._inception_block(256, 128, 128, 192, 32, 96, 32)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = self._inception_block(448, 192, 96, 208, 16, 48, 64)
        self.inception4b = self._inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = self._inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = self._inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = self._inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = self._inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = self._inception_block(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _inception_block(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        return InceptionBlock(in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(InceptionBlock, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5, ch5x5, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1, bias=False),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat([out1, out2, out3, out4], 1)


class InceptionV2Block(nn.Module):
    """Inception V2 Block (with factorized convolutions)"""
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(InceptionV2Block, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )

        # 3x3 卷积分解为 1x3 + 3x1
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3, ch3x3, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )

        # 5x5 卷积分解为 1x5 + 5x1
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5, ch5x5, kernel_size=(5, 1), padding=(2, 0), bias=False),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1, bias=False),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat([out1, out2, out3, out4], 1)


class InceptionV2(nn.Module):
    """Inception V2 model based on original paper"""
    def __init__(self, num_classes=10):
        super(InceptionV2, self).__init__()

        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Inception 模块
        self.inception3a = self._inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = self._inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 分解卷积的 Inception 模块
        self.inception4a = self._inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = self._inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = self._inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = self._inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = self._inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = self._inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = self._inception_block(832, 384, 192, 384, 48, 128, 128)

        # 全局平均池化和分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        # 辅助分类器（用于训练）
        self.aux_classifier = self._create_aux_classifier(512, num_classes)

    def _inception_block(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        return InceptionV2Block(in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj)

    def _create_aux_classifier(self, in_channels, num_classes):
        """创建辅助分类器"""
        return nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(in_channels, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, kernel_size=5, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(768, num_classes)
        )

    def forward(self, x, use_aux=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        
        # 辅助分类器输出
        aux_output = None
        if use_aux and self.training:
            aux_output = self.aux_classifier(x)
        
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        if use_aux and self.training:
            return x, aux_output
        return x


class InceptionV3(nn.Module):
    def __init__(self, num_classes=10):
        super(InceptionV3, self).__init__()

        self.Conv2d_1a_3x3 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.Conv2d_2a_3x3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.Conv2d_2b_3x3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.MaxPool_1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.Conv2d_3b_1x1 = nn.Sequential(
            nn.Conv2d(64, 80, kernel_size=1, bias=False),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True)
        )

        self.Conv2d_4a_3x3 = nn.Sequential(
            nn.Conv2d(80, 192, kernel_size=3, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )

        self.MaxPool_2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)

        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, num_classes)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.MaxPool_1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.MaxPool_2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.branch5x5_1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.branch5x5_2 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.branch3x3dbl_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.branch3x3dbl_2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.branch3x3dbl_3 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(pool_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out1 = self.branch1x1(x)
        out2 = self.branch5x5_1(x)
        out2 = self.branch5x5_2(out2)
        out3 = self.branch3x3dbl_1(x)
        out3 = self.branch3x3dbl_2(out3)
        out3 = self.branch3x3dbl_3(out3)
        out4 = self.branch_pool(x)
        return torch.cat([out1, out2, out3, out4], 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 384, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.branch3x3dbl_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.branch3x3dbl_2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.branch3x3dbl_3 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        out1 = self.branch3x3(x)
        out2 = self.branch3x3dbl_1(x)
        out2 = self.branch3x3dbl_2(out2)
        out2 = self.branch3x3dbl_3(out2)
        out3 = self.branch_pool(x)
        return torch.cat([out1, out2, out3], 1)


class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.branch7x7_1 = nn.Sequential(
            nn.Conv2d(in_channels, channels_7x7, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels_7x7),
            nn.ReLU(inplace=True)
        )
        self.branch7x7_2 = nn.Sequential(
            nn.Conv2d(channels_7x7, channels_7x7, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(channels_7x7),
            nn.ReLU(inplace=True)
        )
        self.branch7x7_3 = nn.Sequential(
            nn.Conv2d(channels_7x7, 192, kernel_size=(7, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.branch7x7dbl_1 = nn.Sequential(
            nn.Conv2d(in_channels, channels_7x7, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels_7x7),
            nn.ReLU(inplace=True)
        )
        self.branch7x7dbl_2 = nn.Sequential(
            nn.Conv2d(channels_7x7, channels_7x7, kernel_size=(7, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(channels_7x7),
            nn.ReLU(inplace=True)
        )
        self.branch7x7dbl_3 = nn.Sequential(
            nn.Conv2d(channels_7x7, channels_7x7, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(channels_7x7),
            nn.ReLU(inplace=True)
        )
        self.branch7x7dbl_4 = nn.Sequential(
            nn.Conv2d(channels_7x7, channels_7x7, kernel_size=(7, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(channels_7x7),
            nn.ReLU(inplace=True)
        )
        self.branch7x7dbl_5 = nn.Sequential(
            nn.Conv2d(channels_7x7, 192, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out1 = self.branch1x1(x)
        out2 = self.branch7x7_1(x)
        out2 = self.branch7x7_2(out2)
        out2 = self.branch7x7_3(out2)
        out3 = self.branch7x7dbl_1(x)
        out3 = self.branch7x7dbl_2(out3)
        out3 = self.branch7x7dbl_3(out3)
        out3 = self.branch7x7dbl_4(out3)
        out3 = self.branch7x7dbl_5(out3)
        out4 = self.branch_pool(x)
        return torch.cat([out1, out2, out3, out4], 1)


class InceptionD(nn.Module):
    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.branch3x3_2 = nn.Sequential(
            nn.Conv2d(192, 320, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True)
        )
        self.branch7x7x3_1 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.branch7x7x3_2 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.branch7x7x3_3 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=(7, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.branch7x7x3_4 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        out1 = self.branch3x3_1(x)
        out1 = self.branch3x3_2(out1)
        out2 = self.branch7x7x3_1(x)
        out2 = self.branch7x7x3_2(out2)
        out2 = self.branch7x7x3_3(out2)
        out2 = self.branch7x7x3_4(out2)
        out3 = self.branch_pool(x)
        return torch.cat([out1, out2, out3], 1)


class InceptionE(nn.Module):
    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 320, kernel_size=1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True)
        )
        self.branch3x3_1 = nn.Sequential(
            nn.Conv2d(in_channels, 384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.branch3x3_2a = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.branch3x3_2b = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.branch3x3dbl_1 = nn.Sequential(
            nn.Conv2d(in_channels, 448, kernel_size=1, bias=False),
            nn.BatchNorm2d(448),
            nn.ReLU(inplace=True)
        )
        self.branch3x3dbl_2 = nn.Sequential(
            nn.Conv2d(448, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.branch3x3dbl_3a = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.branch3x3dbl_3b = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out1 = self.branch1x1(x)
        out2 = self.branch3x3_1(x)
        out2a = self.branch3x3_2a(out2)
        out2b = self.branch3x3_2b(out2)
        out2 = torch.cat([out2a, out2b], 1)
        out3 = self.branch3x3dbl_1(x)
        out3 = self.branch3x3dbl_2(out3)
        out3a = self.branch3x3dbl_3a(out3)
        out3b = self.branch3x3dbl_3b(out3)
        out3 = torch.cat([out3a, out3b], 1)
        out4 = self.branch_pool(x)
        return torch.cat([out1, out2, out3, out4], 1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"运行设备：{device}")

    models = [
        ('InceptionV1', InceptionV1()),
        ('InceptionV2', InceptionV2()),
        ('InceptionV3', InceptionV3()),
    ]

    for name, model in models:
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: 参数量 {total_params:,}")
