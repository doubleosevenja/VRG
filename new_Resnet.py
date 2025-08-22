import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import seaborn as sns

# 导入数据加载器
from dataloader import VrDataImproved, create_dataloaders


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):
    """3D ResNet for video regression - CPU optimized"""

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=1,
                 dropout_rate=0.3):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # 改进的回归头 - 防止过拟合
        self.regressor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(block_inplanes[3] * block.expansion, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # 使用新的回归头
        x = self.regressor(x)

        return x


class VideoRGBFlowFusionModel(nn.Module):
    """RGB + 光流双流融合模型"""

    def __init__(self, model_depth=10, fusion_type='concat', dropout_rate=0.3):
        super().__init__()

        # RGB流 (3通道)
        self.rgb_stream = generate_model(
            model_depth=model_depth,
            n_classes=256,  # 输出特征而不是最终分类
            n_input_channels=3,
            dropout_rate=dropout_rate
        )

        # 光流流 (2通道)
        self.flow_stream = generate_model(
            model_depth=model_depth,
            n_classes=256,  # 输出特征而不是最终分类
            n_input_channels=2,
            dropout_rate=dropout_rate
        )

        # 修改最后的回归头为特征提取器
        self.rgb_stream.regressor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(list(self.rgb_stream.regressor.children())[-3].in_features, 256),
            nn.ReLU()
        )

        self.flow_stream.regressor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(list(self.flow_stream.regressor.children())[-3].in_features, 256),
            nn.ReLU()
        )

        self.fusion_type = fusion_type

        # 融合层
        if fusion_type == 'concat':
            self.fusion = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(512, 128),  # 256 + 256
                nn.ReLU(),
                nn.Dropout(dropout_rate / 2),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        elif fusion_type == 'add':
            self.fusion = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate / 2),
                nn.Linear(64, 1)
            )

    def forward(self, rgb, flow):
        # 分别提取RGB和光流特征
        rgb_features = self.rgb_stream(rgb)  # (batch_size, 256)
        flow_features = self.flow_stream(flow)  # (batch_size, 256)

        # 特征融合
        if self.fusion_type == 'concat':
            combined = torch.cat([rgb_features, flow_features], dim=1)
        elif self.fusion_type == 'add':
            combined = rgb_features + flow_features

        # 最终预测
        output = self.fusion(combined)
        return output


def generate_model(model_depth, **kwargs):
    """生成模型的工厂函数"""
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet3D(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet3D(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet3D(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet3D(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet3D(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet3D(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet3D(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model


class VideoRegressionTrainer:
    """视频回归训练器"""

    def __init__(self, model, train_loader, test_loader, config, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config

        # 损失函数
        self.criterion = nn.MSELoss()
        self.l1_criterion = nn.L1Loss()

        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=config.get('patience', 5),
            factor=0.5,
            verbose=True,
            min_lr=1e-7
        )

        # 早停
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.early_stopping_counter = 0

        # 训练记录
        self.train_losses = []
        self.test_losses = []
        self.test_maes = []
        self.test_r2s = []
        self.best_mae = float('inf')

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, data in enumerate(pbar):
            try:
                # 处理数据 - 假设返回 (rgb, flow, labels) 或 (combined, labels)
                if len(data) == 3:
                    rgb, flow, labels = data
                    rgb = rgb.to(self.device)
                    flow = flow.to(self.device)
                    labels = labels.to(self.device)

                    # 如果是双流模型
                    if hasattr(self.model, 'rgb_stream'):
                        outputs = self.model(rgb, flow)
                    else:
                        # 拼接RGB和光流 (数据是5通道)
                        combined = torch.cat([rgb, flow], dim=1)
                        outputs = self.model(combined)

                elif len(data) == 2:
                    combined, labels = data
                    combined = combined.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(combined)
                else:
                    continue

                if len(labels.shape) == 1:
                    labels = labels.view(-1, 1)

                # 前向传播
                self.optimizer.zero_grad()
                loss = self.criterion(outputs, labels)

                # 反向传播
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.get('grad_clip', 1.0)
                )

                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

            except Exception as e:
                print(f"训练批次 {batch_idx} 出错: {e}")
                continue

        avg_loss = total_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)
        return avg_loss

    def evaluate_epoch(self):
        """评估一个epoch"""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Evaluating')
            for batch_idx, data in enumerate(pbar):
                try:
                    # 处理数据
                    if len(data) == 3:
                        rgb, flow, labels = data
                        rgb = rgb.to(self.device)
                        flow = flow.to(self.device)
                        labels = labels.to(self.device)

                        if hasattr(self.model, 'rgb_stream'):
                            outputs = self.model(rgb, flow)
                        else:
                            combined = torch.cat([rgb, flow], dim=1)
                            outputs = self.model(combined)

                    elif len(data) == 2:
                        combined, labels = data
                        combined = combined.to(self.device)
                        labels = labels.to(self.device)
                        outputs = self.model(combined)
                    else:
                        continue

                    if len(labels.shape) == 1:
                        labels = labels.view(-1, 1)

                    loss = self.criterion(outputs, labels)
                    mae = self.l1_criterion(outputs, labels)

                    total_loss += loss.item()
                    total_mae += mae.item()
                    num_batches += 1

                    # 收集预测结果
                    predictions = outputs.cpu().numpy().flatten()
                    true_labels = labels.cpu().numpy().flatten()

                    all_predictions.extend(predictions)
                    all_labels.extend(true_labels)

                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'MAE': f'{mae.item():.4f}'
                    })

                except Exception as e:
                    print(f"评估批次 {batch_idx} 出错: {e}")
                    continue

        avg_loss = total_loss / max(num_batches, 1)
        avg_mae = total_mae / max(num_batches, 1)

        # 计算R²
        if len(all_labels) > 1:
            r2 = r2_score(all_labels, all_predictions)
        else:
            r2 = 0.0

        self.test_losses.append(avg_loss)
        self.test_maes.append(avg_mae)
        self.test_r2s.append(r2)

        return avg_loss, avg_mae, r2

    def train(self, save_dir='./checkpoints_resnet3d'):
        """完整训练过程"""
        os.makedirs(save_dir, exist_ok=True)

        print(f"=== 开始3D ResNet视频回归训练 ===")
        print(f"设备: {self.device}")
        print(f"配置: {self.config}")
        print(f"训练样本: {len(self.train_loader.dataset)}")
        print(f"测试样本: {len(self.test_loader.dataset)}")

        for epoch in range(self.config['num_epochs']):
            start_time = time.time()

            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            print("-" * 60)

            # 训练
            train_loss = self.train_epoch()

            # 评估
            test_loss, test_mae, test_r2 = self.evaluate_epoch()

            # 学习率调整
            self.scheduler.step(test_loss)

            epoch_time = time.time() - start_time

            # 打印结果
            print(f"训练损失: {train_loss:.4f}")
            print(f"测试损失: {test_loss:.4f}")
            print(f"测试MAE: {test_mae:.4f}")
            print(f"测试R²: {test_r2:.4f}")
            print(f"当前学习率: {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"用时: {epoch_time:.2f}s")

            # 过拟合检测
            if train_loss > 0:
                overfit_ratio = test_loss / train_loss
                if overfit_ratio > 2.0:
                    print(f"⚠️  过拟合警告 (测试/训练损失比: {overfit_ratio:.2f})")

            # 早停检查
            if test_mae < self.best_mae:
                self.best_mae = test_mae
                self.early_stopping_counter = 0
                self.save_checkpoint(epoch, save_dir, is_best=True)
                print(f"✓ 保存最佳模型 (MAE: {test_mae:.4f})")
            else:
                self.early_stopping_counter += 1

            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"早停触发！连续{self.early_stopping_patience}个epoch未改善")
                break

        print("\n=== 训练完成！===")
        print(f"最佳MAE: {self.best_mae:.4f}")

        # 绘制训练曲线
        self.plot_training_curves(save_dir)

    def save_checkpoint(self, epoch, save_dir, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_mae': self.best_mae,
            'config': self.config,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'test_maes': self.test_maes,
            'test_r2s': self.test_r2s
        }

        if is_best:
            torch.save(checkpoint, os.path.join(save_dir, 'best_resnet3d_model.pth'))

        torch.save(checkpoint, os.path.join(save_dir, f'resnet3d_checkpoint_epoch_{epoch + 1}.pth'))

    def plot_training_curves(self, save_dir):
        """绘制训练曲线"""
        plt.figure(figsize=(15, 5))

        # 损失曲线
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # MAE曲线
        plt.subplot(1, 3, 2)
        plt.plot(self.test_maes, label='Test MAE', color='orange')
        plt.title('MAE Curve')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)

        # R²曲线
        plt.subplot(1, 3, 3)
        plt.plot(self.test_r2s, label='Test R²', color='green')
        plt.title('R² Curve')
        plt.xlabel('Epoch')
        plt.ylabel('R²')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves_resnet3d.png'), dpi=300)
        plt.show()


def get_cpu_optimized_config():
    """CPU优化的训练配置"""
    return {
        'batch_size': 2,  # CPU处理3D卷积建议小batch
        'num_workers': 2,
        'num_epochs': 40,
        'lr': 0.0005,  # 稍小的学习率
        'weight_decay': 1e-4,
        'patience': 5,
        'early_stopping_patience': 8,
        'grad_clip': 1.0
    }


def main():
    """主函数"""
    print("=== 3D ResNet视频回归系统 (CPU优化版) ===")

    # 设备检测
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 训练配置
    config = get_cpu_optimized_config()
    print(f"训练配置: {config}")

    try:
        # 创建数据加载器
        print("\n准备数据...")
        train_loader, test_loader = create_dataloaders(
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            use_flow=True  # 使用光流
        )

        print(f"训练集大小: {len(train_loader.dataset)}")
        print(f"测试集大小: {len(test_loader.dataset)}")

        # 创建模型 - 三种选择
        print("\n创建模型...")

        # 选择1: 单流模型 (5通道 = RGB + 光流)
        model = generate_model(
            model_depth=10,  # 使用最轻量的ResNet-10
            n_classes=1,
            n_input_channels=5,  # RGB(3) + 光流(2)
            dropout_rate=0.4
        )

        # 选择2: 双流融合模型 (过拟合问题严重)
        # model = VideoRGBFlowFusionModel(
        #     model_depth=10,
        #     fusion_type='concat',
        #     dropout_rate=0.4
        # )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")

        # 创建训练器
        print("\n创建训练器...")
        trainer = VideoRegressionTrainer(
            model, train_loader, test_loader, config, device
        )

        # 开始训练
        print("\n开始训练...")
        trainer.train()

        print("\n训练完成！检查 ./checkpoints_resnet3d/ 目录")

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


# 测试函数
def test_model_architecture():
    """测试模型架构"""
    print("=== 测试模型架构 ===")

    # 测试单流模型
    model = generate_model(
        model_depth=10,
        n_classes=1,
        n_input_channels=5,  # RGB + 光流
        dropout_rate=0.3
    )

    # 创建测试输入 (batch=1, channels=5, frames=16, height=256, width=256)
    test_input = torch.randn(1, 5, 16, 256, 256)

    model.eval()
    with torch.no_grad():
        output = model(test_input)
        print(f"输入形状: {test_input.shape}")
        print(f"输出形状: {output.shape}")
        print(f"输出值: {output.item():.4f}")

    # 测试双流模型
    print("\n测试双流模型:")
    dual_model = VideoRGBFlowFusionModel(model_depth=10)

    rgb_input = torch.randn(1, 3, 16, 256, 256)
    flow_input = torch.randn(1, 2, 16, 256, 256)

    dual_model.eval()
    with torch.no_grad():
        dual_output = dual_model(rgb_input, flow_input)
        print(f"RGB输入形状: {rgb_input.shape}")
        print(f"光流输入形状: {flow_input.shape}")
        print(f"输出形状: {dual_output.shape}")
        print(f"输出值: {dual_output.item():.4f}")


if __name__ == '__main__':
    # 可以选择运行主训练程序或测试模型
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_model_architecture()
    else:
        main()