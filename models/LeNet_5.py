import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
        )

        # 使用自适应平均池化来处理任意输入尺寸
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier=nn.Sequential(
            nn.Linear(in_features=120 * 7 * 7, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=num_classes)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x=self.features(x)
        x=self.avgpool(x)
        x=torch.flatten(x, 1)
        x=self.classifier(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"运行设备: {device}")

    model=LeNet5(num_classes=10).to(device)
    print("模型实例化成功")

    total_params=sum(p.numel() for p in model.parameters())
    print(f"模型总参数量：{total_params:,}")
    
    # 测试224×224输入（公平对比实验）
    test_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(test_input)
    print(f"测试输入224×224, 输出形状: {output.shape}")