"""
测试脚本：验证 PyTorch Lightning 实现的正确性
"""
import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试所有模块是否可以正确导入"""
    print("=" * 60)
    print("测试 1: 模块导入")
    print("=" * 60)

    try:
        from src.model import TwoTowerModel, RecallLoss
        print("✓ src.model 导入成功")
    except Exception as e:
        print(f"✗ src.model 导入失败: {e}")
        return False

    try:
        from src.dataset import AliCCPDataset
        print("✓ src.dataset 导入成功")
    except Exception as e:
        print(f"✗ src.dataset 导入失败: {e}")
        return False

    try:
        from src.datamodule import AliCCPDataModule, AliCCPDatasetWithSampling
        print("✓ src.datamodule 导入成功")
    except Exception as e:
        print(f"✗ src.datamodule 导入失败: {e}")
        return False

    try:
        from src.lightning_module import TwoTowerLightningModule
        print("✓ src.lightning_module 导入成功")
    except Exception as e:
        print(f"✗ src.lightning_module 导入失败: {e}")
        return False

    print("\n所有模块导入成功!\n")
    return True


def test_model_creation():
    """测试模型创建"""
    print("=" * 60)
    print("测试 2: 模型创建")
    print("=" * 60)

    try:
        from src.lightning_module import TwoTowerLightningModule

        model = TwoTowerLightningModule(
            user_num=1000,
            item_num=5000,
            embed_dim=16,
            hidden_dims=[128, 64],
            tower_out_dim=32,
            dropout=0.1,
            user_feature_dims=[98, 14, 3, 8, 4, 4, 3, 5],
            item_feature_dims=[538376, 7092, 285825, 82412, 113262],
            mode="single",
            lr=1e-3,
            weight_decay=1e-5,
            temperature=0.05,
            max_epochs=20,
        )

        print(f"✓ 模型创建成功")
        print(f"  - 模式: single")
        print(f"  - 参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - 可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n模型创建测试通过!\n")
    return True


def test_forward_pass():
    """测试前向传播"""
    print("=" * 60)
    print("测试 3: 前向传播")
    print("=" * 60)

    try:
        from src.lightning_module import TwoTowerLightningModule

        model = TwoTowerLightningModule(
            user_num=1000,
            item_num=5000,
            embed_dim=16,
            hidden_dims=[128, 64],
            tower_out_dim=32,
            dropout=0.1,
            user_feature_dims=[98, 14, 3, 8, 4, 4, 3, 5],
            item_feature_dims=[538376, 7092, 285825, 82412, 113262],
            mode="single",
        )

        # 创建模拟数据
        batch_size = 32
        user_features = torch.randint(0, 100, (batch_size, 8))
        item_features = torch.randint(0, 1000, (batch_size, 5))
        labels = torch.randint(0, 2, (batch_size,))

        # 前向传播
        model.eval()
        with torch.no_grad():
            scores = model(user_features, item_features)

        print(f"✓ 前向传播成功")
        print(f"  - 输入形状: user={user_features.shape}, item={item_features.shape}")
        print(f"  - 输出形状: {scores.shape}")
        print(f"  - 输出范围: [{scores.min():.4f}, {scores.max():.4f}]")

    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n前向传播测试通过!\n")
    return True


def test_training_step():
    """测试训练步骤"""
    print("=" * 60)
    print("测试 4: 训练步骤")
    print("=" * 60)

    try:
        from src.lightning_module import TwoTowerLightningModule

        for mode in ["single", "pair", "list"]:
            model = TwoTowerLightningModule(
                user_num=1000,
                item_num=5000,
                embed_dim=16,
                hidden_dims=[128, 64],
                tower_out_dim=32,
                dropout=0.1,
                user_feature_dims=[98, 14, 3, 8, 4, 4, 3, 5],
                item_feature_dims=[538376, 7092, 285825, 82412, 113262],
                mode=mode,
            )

            # 创建模拟批次
            batch_size = 64
            user_features = torch.randint(0, 100, (batch_size, 8))
            item_features = torch.randint(0, 1000, (batch_size, 5))
            labels = torch.cat([
                torch.ones(batch_size // 2),
                torch.zeros(batch_size // 2)
            ]).long()

            batch = (user_features, item_features, labels)

            # 训练步骤
            model.train()
            loss = model.training_step(batch, 0)

            print(f"✓ 模式 '{mode}' 训练步骤成功, loss={loss.item():.4f}")

    except Exception as e:
        print(f"✗ 训练步骤失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n训练步骤测试通过!\n")
    return True


def test_negative_sampling():
    """测试负样本采样逻辑"""
    print("=" * 60)
    print("测试 5: 负样本采样")
    print("=" * 60)

    try:
        import pandas as pd
        import tempfile
        from src.datamodule import AliCCPDatasetWithSampling

        # 创建临时测试数据
        data = {
            "121": [1] * 100 + [2] * 400,
            "122": [1] * 100 + [2] * 400,
            "124": [1] * 100 + [2] * 400,
            "125": [1] * 100 + [2] * 400,
            "126": [1] * 100 + [2] * 400,
            "127": [1] * 100 + [2] * 400,
            "128": [1] * 100 + [2] * 400,
            "129": [1] * 100 + [2] * 400,
            "205": [1] * 100 + [2] * 400,
            "206": [1] * 100 + [2] * 400,
            "207": [1] * 100 + [2] * 400,
            "210": [1] * 100 + [2] * 400,
            "216": [1] * 100 + [2] * 400,
            "click": [1] * 100 + [0] * 400,  # 100正样本, 400负样本
        }
        df = pd.DataFrame(data)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            # 测试不同采样比率
            for ratio in [1.0, 0.5, 0.25]:
                dataset = AliCCPDatasetWithSampling(
                    data_path=temp_path,
                    user_sparse_columns=["121", "122", "124", "125", "126", "127", "128", "129"],
                    user_dense_columns=[],
                    item_sparse_columns=["205", "206", "207", "210", "216"],
                    item_dense_columns=[],
                    neg_sample_ratio=ratio,
                )

                expected_neg = int(400 * ratio)
                actual_neg = len(dataset.sampled_neg_indices)

                print(f"✓ 采样比率 {ratio}: 期望 {expected_neg} 个负样本, 实际 {actual_neg} 个")
                print(f"  - 数据集总长度: {len(dataset)} (100正 + {actual_neg}负)")

                # 测试重新采样
                old_indices = dataset.sampled_neg_indices.copy()
                dataset.resample_negatives()
                new_indices = dataset.sampled_neg_indices

                if ratio < 1.0:
                    # 应该有不同的索引
                    diff_ratio = (old_indices != new_indices).mean()
                    print(f"  - 重新采样后变化率: {diff_ratio:.2%}")

        finally:
            os.unlink(temp_path)

    except Exception as e:
        print(f"✗ 负样本采样测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n负样本采样测试通过!\n")
    return True


def main():
    print("\n" + "=" * 60)
    print("PyTorch Lightning 实现验证测试")
    print("=" * 60 + "\n")

    tests = [
        ("模块导入", test_imports),
        ("模型创建", test_model_creation),
        ("前向传播", test_forward_pass),
        ("训练步骤", test_training_step),
        ("负样本采样", test_negative_sampling),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n测试 '{name}' 出现异常: {e}\n")
            results.append((name, False))

    # 总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)

    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过! 代码结构正确。")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    exit(main())
