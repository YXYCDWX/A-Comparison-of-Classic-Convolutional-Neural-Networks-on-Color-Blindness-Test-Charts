import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        return torch.cat([x, out], 1)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        out = self.pool(out)
        return out


class DenseNet121(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNet121, self).__init__()
        self.growth_rate = 32
        self.num_init_features = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        num_features = 64
        self.dense1 = self._make_dense_block(num_features, 6)
        num_features += 6 * self.growth_rate
        
        self.trans1 = self._make_transition(num_features, num_features // 2)
        num_features = num_features // 2
        
        self.dense2 = self._make_dense_block(num_features, 12)
        num_features += 12 * self.growth_rate
        
        self.trans2 = self._make_transition(num_features, num_features // 2)
        num_features = num_features // 2
        
        self.dense3 = self._make_dense_block(num_features, 24)
        num_features += 24 * self.growth_rate
        
        self.trans3 = self._make_transition(num_features, num_features // 2)
        num_features = num_features // 2
        
        self.dense4 = self._make_dense_block(num_features, 16)
        num_features += 16 * self.growth_rate

        self.bn = nn.BatchNorm2d(num_features)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, num_classes)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_dense_block(self, in_channels, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(Bottleneck(in_channels + i * self.growth_rate, self.growth_rate))
        return nn.Sequential(*layers)

    def _make_transition(self, in_channels, out_channels):
        return Transition(in_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.trans3(x)
        x = self.dense4(x)
        
        x = self.bn(x)
        x = self.relu_final(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class DenseNet169(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNet169, self).__init__()
        self.growth_rate = 32
        self.num_init_features = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        num_features = 64
        self.dense1 = self._make_dense_block(num_features, 6)
        num_features += 6 * self.growth_rate
        
        self.trans1 = self._make_transition(num_features, num_features // 2)
        num_features = num_features // 2
        
        self.dense2 = self._make_dense_block(num_features, 12)
        num_features += 12 * self.growth_rate
        
        self.trans2 = self._make_transition(num_features, num_features // 2)
        num_features = num_features // 2
        
        self.dense3 = self._make_dense_block(num_features, 32)
        num_features += 32 * self.growth_rate
        
        self.trans3 = self._make_transition(num_features, num_features // 2)
        num_features = num_features // 2
        
        self.dense4 = self._make_dense_block(num_features, 32)
        num_features += 32 * self.growth_rate

        self.bn = nn.BatchNorm2d(num_features)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, num_classes)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_dense_block(self, in_channels, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(Bottleneck(in_channels + i * self.growth_rate, self.growth_rate))
        return nn.Sequential(*layers)

    def _make_transition(self, in_channels, out_channels):
        return Transition(in_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.trans3(x)
        x = self.dense4(x)
        
        x = self.bn(x)
        x = self.relu_final(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"运行设备：{device}")

    models = [
        ('DenseNet121', DenseNet121()),
        ('DenseNet169', DenseNet169()),
    ]

    for name, model in models:
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: 参数量 {total_params:,}")
