import argparse

from finetune_best_pika_model_v4 import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="M11 improved fine-tune from M10 using lighter class weight and higher dropout"
    )

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, default="")

    parser.add_argument("--base_checkpoint", type=str, required=True)

    parser.add_argument("--data_root", type=str, default="/content/processed_pika_best")
    parser.add_argument("--graph_labels_json", type=str, default="")
    parser.add_argument("--graph_pmi_npy", type=str, default="")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/model/M11_sqrt035_dropout045",
    )
    parser.add_argument(
        "--best_name",
        type=str,
        default="M11_sqrt035_dropout045_best.pth",
    )
    parser.add_argument(
        "--last_name",
        type=str,
        default="M11_sqrt035_dropout045_last.pth",
    )

    parser.add_argument("--pill_model_name", type=str, default="tf_efficientnetv2_s.in21k_ft_in1k")
    parser.add_argument("--pres_model_name", type=str, default="resnet18.a1_in1k")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--max_context_len", type=int, default=5)

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)

    # M11 improved hyperparameters
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.03)
    parser.add_argument("--class_weight_exponent", type=float, default=0.35)
    parser.add_argument("--dropout_p", type=float, default=0.45)
    parser.add_argument("--patience", type=int, default=3)

    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
