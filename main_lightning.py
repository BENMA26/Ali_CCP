import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.datamodule import AliCCPDataModule
from src.lightning_module import TwoTowerLightningModule


def parse_args():
    parser = argparse.ArgumentParser(description="Ali CCP Two-Tower Model Training with PyTorch Lightning")

    # 数据路径参数
    parser.add_argument("--train_data_path", type=str,
                        default="/work/home/maben/project/rec_sys/projects/Ali_CCP/datasets/datasetsali_ccp_train.csv",
                        help="训练数据路径")
    parser.add_argument("--val_data_path", type=str,
                        default="/work/home/maben/project/rec_sys/projects/Ali_CCP/datasets/datasetsali_ccp_val.csv",
                        help="验证数据路径")

    # 模型参数
    parser.add_argument("--user_num", type=int, default=1000, help="用户数量")
    parser.add_argument("--item_num", type=int, default=5000, help="物品数量")
    parser.add_argument("--embed_dim", type=int, default=16, help="嵌入维度")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 64], help="隐藏层维度列表")
    parser.add_argument("--tower_out_dim", type=int, default=32, help="塔输出维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout比率")

    # 训练参数
    parser.add_argument("--mode", type=str, default="single", choices=["single", "pair", "list"],
                        help="训练模式: single(Pointwise BCE), pair(Pairwise BPR), list(Listwise InfoNCE)")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2048, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--temperature", type=float, default=0.05, help="InfoNCE温度参数(仅list模式)")

    # 负样本采样参数
    parser.add_argument("--neg_sample_ratio", type=float, default=1.0,
                        help="负样本采样比率 (0.0-1.0)，1.0表示使用全部负样本，0.5表示使用50%%负样本")

    # 数据加载参数
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")

    # 保存和日志参数
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="模型保存目录")
    parser.add_argument("--log_dir", type=str, default="runs", help="日志保存目录")
    parser.add_argument("--save_top_k", type=int, default=3, help="保存最好的k个模型")

    # Early stopping参数
    parser.add_argument("--early_stop_patience", type=int, default=5, help="Early stopping patience，-1表示关闭")

    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从检查点恢复训练")
    parser.add_argument("--accelerator", type=str, default="auto", help="加速器类型: auto, cpu, gpu, tpu")
    parser.add_argument("--devices", type=int, default=1, help="使用的设备数量")
    parser.add_argument("--precision", type=str, default="32", help="训练精度: 32, 16, bf16")

    return parser.parse_args()


def main():
    args = parse_args()

    # 设置随机种子
    pl.seed_everything(args.seed)

    # 特征列定义
    USER_SPARSE = ["121", "122", "124", "125", "126", "127", "128", "129"]
    USER_DENSE = []
    ITEM_SPARSE = ["205", "206", "207", "210", "216"]
    ITEM_DENSE = []

    # 特征维度（vocab size）
    user_feature_dims = [98, 14, 3, 8, 4, 4, 3, 5]
    item_feature_dims = [538376, 7092, 285825, 82412, 113262]

    # 创建数据模块
    datamodule = AliCCPDataModule(
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        user_sparse_columns=USER_SPARSE,
        user_dense_columns=USER_DENSE,
        item_sparse_columns=ITEM_SPARSE,
        item_dense_columns=ITEM_DENSE,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        neg_sample_ratio=args.neg_sample_ratio,
    )

    # 创建模型
    model = TwoTowerLightningModule(
        user_num=args.user_num,
        item_num=args.item_num,
        embed_dim=args.embed_dim,
        hidden_dims=args.hidden_dims,
        tower_out_dim=args.tower_out_dim,
        dropout=args.dropout,
        user_feature_dims=user_feature_dims,
        item_feature_dims=item_feature_dims,
        mode=args.mode,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        max_epochs=args.epochs,
    )

    # 创建回调函数
    callbacks = []

    # ModelCheckpoint - 保存最好的模型
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename=f"{args.mode}-{{epoch:02d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=args.save_top_k,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Early Stopping
    if args.early_stop_patience > 0:
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=args.early_stop_patience,
            mode="min",
            verbose=True,
        )
        callbacks.append(early_stop_callback)

    # Learning Rate Monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    # 创建日志记录器
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=f"{args.mode}_model",
    )

    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # 开始训练
    print("\n" + "=" * 80)
    print(f"开始训练 - 模式: {args.mode}")
    print(f"负样本采样比率: {args.neg_sample_ratio}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.epochs}")
    print("=" * 80 + "\n")

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=args.resume_from_checkpoint,
    )

    # 训练完成
    print("\n" + "=" * 80)
    print("训练完成!")
    print(f"最佳模型保存在: {checkpoint_callback.best_model_path}")
    print(f"最佳验证损失: {checkpoint_callback.best_model_score:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
