"""
CNN 模型训练与预测

提供三个主要接口：
- train_model(): 从 h5 文件训练分类模型
- predict(): 对新样本做推理预测
- cross_validate(): 小样本场景下的交叉验证
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

from .model import MzrtCNN
from .dataset import DynamicSparseH5Dataset


def _get_device(device=None):
    """自动检测最佳设备"""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train_model(
    h5_path,
    target_covariate='class',
    rt_precision=3.0,
    mz_precision=1.0,
    num_epochs=50,
    batch_size=8,
    learning_rate=1e-3,
    val_ratio=0.2,
    base_filters=32,
    dropout=0.5,
    augment=True,
    aug_rt_shift_s=3.0,
    aug_mz_shift_ppm=10.0,
    save_path=None,
    device=None,
    **dataset_kwargs,
):
    """从 h5 文件训练 CNN 分类模型

    Args:
        h5_path: mzrt2h5 生成的 HDF5 文件路径
        target_covariate: 分类目标（h5 文件中的 covariate 名称）
        rt_precision: 目标 RT 分辨率（秒）
        mz_precision: 目标 m/z 分辨率（Da）
        num_epochs: 训练轮数
        batch_size: 批大小
        learning_rate: 学习率
        val_ratio: 验证集比例
        base_filters: CNN 基础通道数
        dropout: Dropout 比例
        augment: 是否启用数据增强
        aug_rt_shift_s: RT 偏移增强范围（秒）
        aug_mz_shift_ppm: m/z 偏移增强范围（ppm）
        save_path: 模型保存路径（None 则不保存）
        device: 运行设备
        **dataset_kwargs: 传递给 DynamicSparseH5Dataset 的额外参数
            （如 covariate_filters, crop_rt_range, crop_mz_range, min_intensity）

    Returns:
        dict: {
            'model': 训练好的模型,
            'history': {'train_loss': [...], 'train_acc': [...],
                        'val_loss': [...], 'val_acc': [...]},
            'best_val_acc': 最佳验证准确率,
            'num_classes': 类别数,
            'config': 训练配置（用于复现）
        }
    """
    device = _get_device(device)
    print(f"Device: {device}")

    # 数据集（训练集启用增强，验证集不启用）
    train_ds = DynamicSparseH5Dataset(
        h5_path, rt_precision, mz_precision,
        target_covariate=target_covariate,
        augment=augment,
        aug_rt_shift_s=aug_rt_shift_s,
        aug_mz_shift_ppm=aug_mz_shift_ppm,
        apply_log1p_norm=True,
        **dataset_kwargs,
    )
    val_ds = DynamicSparseH5Dataset(
        h5_path, rt_precision, mz_precision,
        target_covariate=target_covariate,
        augment=False,
        apply_log1p_norm=True,
        **dataset_kwargs,
    )

    n = len(train_ds)
    indices = np.random.permutation(n)
    n_val = max(1, int(n * val_ratio))
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=batch_size,
                            shuffle=False, num_workers=0)

    # 自动检测类别数
    labels = []
    for i in range(n):
        _, label = val_ds[i]
        labels.append(label.item())
    num_classes = len(set(labels))
    print(f"Samples: {n} ({n - n_val} train, {n_val} val), Classes: {num_classes}")

    # 模型
    model = MzrtCNN(num_classes, base_filters=base_filters, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        # 训练
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels_batch in train_loader:
            images, labels_batch = images.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels_batch).sum().item()
            total += images.size(0)
        train_loss = total_loss / total
        train_acc = correct / total

        # 验证
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels_batch in val_loader:
                images, labels_batch = images.to(device), labels_batch.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels_batch)
                total_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels_batch).sum().item()
                total += images.size(0)
        val_loss = total_loss / total if total > 0 else 0
        val_acc = correct / total if total > 0 else 0

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    # 恢复最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    print(f"\nBest val accuracy: {best_val_acc:.3f}")

    config = {
        'rt_precision': rt_precision,
        'mz_precision': mz_precision,
        'target_covariate': target_covariate,
        'num_classes': num_classes,
        'base_filters': base_filters,
        'dropout': dropout,
    }

    if save_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
        }, save_path)
        print(f"Model saved to {save_path}")

    return {
        'model': model,
        'history': history,
        'best_val_acc': best_val_acc,
        'num_classes': num_classes,
        'config': config,
    }


def predict(
    h5_path,
    model_path,
    rt_precision=None,
    mz_precision=None,
    batch_size=8,
    device=None,
    **dataset_kwargs,
):
    """对新样本做推理预测

    Args:
        h5_path: 待预测的 HDF5 文件路径
        model_path: 训练好的 .pth 模型路径
        rt_precision: RT 分辨率（None 则从模型配置读取）
        mz_precision: m/z 分辨率（None 则从模型配置读取）
        batch_size: 批大小
        device: 运行设备
        **dataset_kwargs: 传递给 DynamicSparseH5Dataset 的额外参数

    Returns:
        dict: {
            'predictions': np.ndarray 预测类别,
            'probabilities': np.ndarray 各类概率,
            'sample_ids': list 样本 ID,
        }
    """
    device = _get_device(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    if rt_precision is None:
        rt_precision = config['rt_precision']
    if mz_precision is None:
        mz_precision = config['mz_precision']

    model = MzrtCNN(
        config['num_classes'],
        base_filters=config.get('base_filters', 32),
        dropout=0.0,  # 推理时关闭 dropout
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataset = DynamicSparseH5Dataset(
        h5_path, rt_precision, mz_precision,
        target_covariate=None,  # 推理模式
        augment=False,
        apply_log1p_norm=True,
        **dataset_kwargs,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    predictions = all_probs.argmax(axis=1)

    # 读取样本 ID
    import h5py
    with h5py.File(h5_path, 'r') as f:
        if 'sample_id' in f:
            sample_ids = [s.decode() if isinstance(s, bytes) else str(s)
                          for s in f['sample_id'][:]]
        else:
            sample_ids = [f"sample_{i}" for i in range(len(predictions))]

    return {
        'predictions': predictions,
        'probabilities': all_probs,
        'sample_ids': sample_ids[:len(predictions)],
    }


def cross_validate(
    h5_path,
    target_covariate='class',
    rt_precision=3.0,
    mz_precision=1.0,
    n_folds=5,
    num_epochs=50,
    batch_size=8,
    learning_rate=1e-3,
    base_filters=32,
    dropout=0.5,
    device=None,
    **dataset_kwargs,
):
    """K 折交叉验证（适合小样本场景）

    Args:
        h5_path: HDF5 文件路径
        target_covariate: 分类目标
        n_folds: 折数
        其他参数同 train_model

    Returns:
        dict: {
            'fold_accs': list 各折准确率,
            'mean_acc': 平均准确率,
            'std_acc': 准确率标准差,
        }
    """
    device = _get_device(device)

    dataset = DynamicSparseH5Dataset(
        h5_path, rt_precision, mz_precision,
        target_covariate=target_covariate,
        augment=False,
        apply_log1p_norm=True,
        **dataset_kwargs,
    )

    n = len(dataset)
    indices = np.random.permutation(n)
    fold_size = n // n_folds

    # 检测类别数
    labels = []
    for i in range(n):
        _, label = dataset[i]
        labels.append(label.item())
    num_classes = len(set(labels))
    print(f"Cross-validation: {n} samples, {num_classes} classes, {n_folds} folds")

    # 带增强的训练集
    train_ds = DynamicSparseH5Dataset(
        h5_path, rt_precision, mz_precision,
        target_covariate=target_covariate,
        augment=True,
        aug_rt_shift_s=3.0,
        aug_mz_shift_ppm=10.0,
        apply_log1p_norm=True,
        **dataset_kwargs,
    )

    fold_accs = []
    criterion = nn.CrossEntropyLoss()

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n
        val_idx = indices[val_start:val_end].tolist()
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]]).tolist()

        train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=batch_size,
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size,
                                shuffle=False, num_workers=0)

        model = MzrtCNN(num_classes, base_filters=base_filters, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        best_acc = 0.0
        for epoch in range(num_epochs):
            model.train()
            for images, labels_batch in train_loader:
                images, labels_batch = images.to(device), labels_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels_batch in val_loader:
                    images, labels_batch = images.to(device), labels_batch.to(device)
                    correct += (model(images).argmax(1) == labels_batch).sum().item()
                    total += images.size(0)
            acc = correct / total if total > 0 else 0
            best_acc = max(best_acc, acc)

        fold_accs.append(best_acc)
        print(f"Fold {fold+1}/{n_folds}: accuracy = {best_acc:.3f}")

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    print(f"\nCV Result: {mean_acc:.3f} ± {std_acc:.3f}")

    return {
        'fold_accs': fold_accs,
        'mean_acc': mean_acc,
        'std_acc': std_acc,
    }
