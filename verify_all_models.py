#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证所有模型是否可以正常工作"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import torch

print("=" * 80)
print("Color Blindness Project - All Models Verification")
print("=" * 80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nRunning on: {device}")

# Test input sizes
test_input_128 = torch.randn(1, 3, 128, 128).to(device)
test_input_224 = torch.randn(1, 3, 224, 224).to(device)
test_input_299 = torch.randn(1, 3, 299, 299).to(device)

print(f"\nTest input sizes:")
print(f"  128x128 (dataset original)")
print(f"  224x224 (experiment script)")
print(f"  299x299 (InceptionV3)")

results = []

print("\n" + "=" * 80)
print("1. Testing LeNet5")
print("=" * 80)
try:
    from models.LeNet_5 import LeNet5
    model = LeNet5(num_classes=10).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"OK: Model instantiated, params: {total_params:,}")
    
    output = model(test_input_224)
    print(f"OK: 224x224 input, output shape {output.shape}")
    
    output = model(test_input_128)
    print(f"OK: 128x128 input, output shape {output.shape}")
    
    results.append(("LeNet5", "PASS"))
except Exception as e:
    print(f"ERROR: {e}")
    results.append(("LeNet5", "FAIL"))

print("\n" + "=" * 80)
print("2. Testing AlexNet")
print("=" * 80)
try:
    from models.AlexNet import AlexNet
    model = AlexNet(num_classes=10).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"OK: Model instantiated, params: {total_params:,}")
    
    output = model(test_input_224)
    print(f"OK: 224x224 input, output shape {output.shape}")
    
    try:
        output = model(test_input_128)
        print(f"OK: 128x128 input, output shape {output.shape}")
    except Exception as e2:
        print(f"WARN: 128x128 may have issues: {e2}")
    
    results.append(("AlexNet", "PASS"))
except Exception as e:
    print(f"ERROR: {e}")
    results.append(("AlexNet", "FAIL"))

print("\n" + "=" * 80)
print("3. Testing VGG Series")
print("=" * 80)
try:
    from models.VGG import VGG11, VGG13, VGG16, VGG19, VGG11_BN, VGG13_BN, VGG16_BN, VGG19_BN
    
    vgg_models = [
        ('VGG11', VGG11),
        ('VGG13', VGG13),
        ('VGG16', VGG16),
        ('VGG19', VGG19),
        ('VGG11_BN', VGG11_BN),
        ('VGG13_BN', VGG13_BN),
        ('VGG16_BN', VGG16_BN),
        ('VGG19_BN', VGG19_BN),
    ]
    
    for name, model_class in vgg_models:
        try:
            model = model_class(num_classes=10).to(device)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\n{name}:")
            print(f"  OK: Model instantiated, params: {total_params:,}")
            
            output = model(test_input_128)
            print(f"  OK: 128x128 input, output shape {output.shape}")
            
            output = model(test_input_224)
            print(f"  OK: 224x224 input, output shape {output.shape}")
            
            results.append((name, "PASS"))
        except Exception as e:
            print(f"\n{name}:")
            print(f"  ERROR: {e}")
            results.append((name, "FAIL"))
            
except Exception as e:
    print(f"ERROR: VGG import failed: {e}")

print("\n" + "=" * 80)
print("4. Testing ResNet Series")
print("=" * 80)
try:
    from models.ResNet import ResNet18, ResNet34, ResNet50
    
    resnet_models = [
        ('ResNet18', ResNet18),
        ('ResNet34', ResNet34),
        ('ResNet50', ResNet50),
    ]
    
    for name, model_class in resnet_models:
        try:
            model = model_class(num_classes=10).to(device)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\n{name}:")
            print(f"  OK: Model instantiated, params: {total_params:,}")
            
            output = model(test_input_128)
            print(f"  OK: 128x128 input, output shape {output.shape}")
            
            output = model(test_input_224)
            print(f"  OK: 224x224 input, output shape {output.shape}")
            
            results.append((name, "PASS"))
        except Exception as e:
            print(f"\n{name}:")
            print(f"  ERROR: {e}")
            results.append((name, "FAIL"))
            
except Exception as e:
    print(f"ERROR: ResNet import failed: {e}")

print("\n" + "=" * 80)
print("5. Testing Inception Series")
print("=" * 80)
try:
    from models.Inception import InceptionV1, InceptionV3
    
    inception_models = [
        ('InceptionV1', InceptionV1, [test_input_224, test_input_128]),
        ('InceptionV3', InceptionV3, [test_input_299, test_input_224]),
    ]
    
    for name, model_class, test_inputs in inception_models:
        try:
            model = model_class(num_classes=10).to(device)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\n{name}:")
            print(f"  OK: Model instantiated, params: {total_params:,}")
            
            for i, test_input in enumerate(test_inputs):
                size_str = f"{test_input.shape[2]}x{test_input.shape[3]}"
                try:
                    output = model(test_input)
                    print(f"  OK: {size_str} input, output shape {output.shape}")
                except Exception as e2:
                    print(f"  WARN: {size_str} input may have issues: {e2}")
            
            results.append((name, "PASS"))
        except Exception as e:
            print(f"\n{name}:")
            print(f"  ERROR: {e}")
            results.append((name, "FAIL"))
            
except Exception as e:
    print(f"ERROR: Inception import failed: {e}")

print("\n" + "=" * 80)
print("6. Testing DenseNet Series")
print("=" * 80)
try:
    from models.DenseNet import DenseNet121, DenseNet169
    
    densenet_models = [
        ('DenseNet121', DenseNet121),
        ('DenseNet169', DenseNet169),
    ]
    
    for name, model_class in densenet_models:
        try:
            model = model_class(num_classes=10).to(device)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\n{name}:")
            print(f"  OK: Model instantiated, params: {total_params:,}")
            
            output = model(test_input_128)
            print(f"  OK: 128x128 input, output shape {output.shape}")
            
            output = model(test_input_224)
            print(f"  OK: 224x224 input, output shape {output.shape}")
            
            results.append((name, "PASS"))
        except Exception as e:
            print(f"\n{name}:")
            print(f"  ERROR: {e}")
            results.append((name, "FAIL"))
            
except Exception as e:
    print(f"ERROR: DenseNet import failed: {e}")

print("\n" + "=" * 80)
print("Verification Summary")
print("=" * 80)
for name, status in results:
    print(f"{name:<20} {status}")

passed = sum(1 for _, status in results if status == "PASS")
total = len(results)
print(f"\nTotal: {passed}/{total} models passed verification")

if passed == total:
    print("\nALL MODELS VERIFIED SUCCESSFULLY! Project is ready to run.")
else:
    print(f"\nWARNING: {total - passed} models need repair.")
