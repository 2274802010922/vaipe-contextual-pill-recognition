import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class ModelSpec:
    key: str
    display_name: str
    family: str
    checkpoint_filename: str

    # thư mục chứa artifact riêng của model
    artifact_subdir: str

    # đặc điểm model
    needs_context: bool = False
    needs_graph: bool = False
    needs_prescription_branch: bool = False

    # backbone / ghi chú
    pill_backbone: Optional[str] = None
    prescription_backbone: Optional[str] = None
    image_size: int = 224
    notes: str = ""

    # đường dẫn được resolve sau
    checkpoint_path: Optional[str] = None
    artifact_dir: Optional[str] = None
    idx_to_label_path: Optional[str] = None
    class_names_path: Optional[str] = None
    graph_labels_path: Optional[str] = None
    graph_pmi_path: Optional[str] = None
    extra_config_path: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    def expected_files(self) -> List[str]:
        paths = []
        if self.checkpoint_path is not None:
            paths.append(self.checkpoint_path)

        # Tất cả model đều nên có ít nhất 1 file map output_idx -> original_label
        if self.idx_to_label_path is not None:
            paths.append(self.idx_to_label_path)

        # File class names là tùy chọn nhưng nên có
        if self.class_names_path is not None:
            paths.append(self.class_names_path)

        if self.needs_graph:
            if self.graph_labels_path is not None:
                paths.append(self.graph_labels_path)
            if self.graph_pmi_path is not None:
                paths.append(self.graph_pmi_path)

        if self.extra_config_path is not None:
            paths.append(self.extra_config_path)

        return paths


def resolve_model_spec(
    spec: ModelSpec,
    model_dir: str,
    artifact_root: str,
) -> ModelSpec:
    artifact_dir = os.path.join(artifact_root, spec.artifact_subdir)

    return ModelSpec(
        key=spec.key,
        display_name=spec.display_name,
        family=spec.family,
        checkpoint_filename=spec.checkpoint_filename,
        artifact_subdir=spec.artifact_subdir,
        needs_context=spec.needs_context,
        needs_graph=spec.needs_graph,
        needs_prescription_branch=spec.needs_prescription_branch,
        pill_backbone=spec.pill_backbone,
        prescription_backbone=spec.prescription_backbone,
        image_size=spec.image_size,
        notes=spec.notes,
        checkpoint_path=os.path.join(model_dir, spec.checkpoint_filename),
        artifact_dir=artifact_dir,
        idx_to_label_path=os.path.join(artifact_dir, "idx_to_label.json"),
        class_names_path=os.path.join(artifact_dir, "class_names.json"),
        graph_labels_path=os.path.join(artifact_dir, "graph_labels.json"),
        graph_pmi_path=os.path.join(artifact_dir, "graph_pmi.npy"),
        extra_config_path=os.path.join(artifact_dir, "model_config.json"),
    )


def build_model_registry(
    model_dir: str,
    artifact_root: str,
) -> Dict[str, ModelSpec]:
    """
    model_dir:
        thư mục chứa 9 file .pth, ví dụ:
        /content/drive/MyDrive/VAIPE_project/model

    artifact_root:
        thư mục chứa artifact cho từng model, ví dụ:
        /content/drive/MyDrive/VAIPE_project/model_artifacts
    """

    base_specs = [
        ModelSpec(
            key="M1",
            display_name="M1 Baseline",
            family="baseline_image_only",
            checkpoint_filename="M1_baseline_best.pth",
            artifact_subdir="M1_baseline",
            needs_context=False,
            needs_graph=False,
            needs_prescription_branch=False,
            pill_backbone="tf_efficientnetv2_s.in21k_ft_in1k",
            notes="Baseline chỉ dùng ảnh crop viên thuốc.",
        ),
        ModelSpec(
            key="M2",
            display_name="M2 PIKA-like v1",
            family="pika_like_v1",
            checkpoint_filename="M2_pika_like_v1_best.pth",
            artifact_subdir="M2_pika_like_v1",
            needs_context=True,
            needs_graph=False,
            needs_prescription_branch=True,
            pill_backbone="tf_efficientnetv2_s.in21k_ft_in1k",
            prescription_backbone="resnet18.a1_in1k",
            notes="Bản context đầu tiên theo hướng PIKA-like.",
        ),
        ModelSpec(
            key="M3",
            display_name="M3 PIKA v2",
            family="pika_v2_context_labels",
            checkpoint_filename="M3_pika_v2_best.pth",
            artifact_subdir="M3_pika_v2",
            needs_context=True,
            needs_graph=False,
            needs_prescription_branch=True,
            pill_backbone="tf_efficientnetv2_s.in21k_ft_in1k",
            prescription_backbone="resnet18.a1_in1k",
            notes="Bản context labels v2.",
        ),
        ModelSpec(
            key="M4",
            display_name="M4 PIKA v3",
            family="pika_v3_triple_context",
            checkpoint_filename="M4_pika_v3_best.pth",
            artifact_subdir="M4_pika_v3",
            needs_context=True,
            needs_graph=False,
            needs_prescription_branch=True,
            pill_backbone="tf_efficientnetv2_s.in21k_ft_in1k",
            prescription_backbone="resnet18.a1_in1k",
            notes="Bản triple context v3.",
        ),
        ModelSpec(
            key="M5",
            display_name="M5 PIKA graph v1",
            family="pika_graph_v1",
            checkpoint_filename="M5_pika_graph_best.pth",
            artifact_subdir="M5_pika_graph_v1",
            needs_context=True,
            needs_graph=True,
            needs_prescription_branch=True,
            pill_backbone="tf_efficientnetv2_s.in21k_ft_in1k",
            prescription_backbone="resnet18.a1_in1k",
            notes="Bản graph context đầu tiên.",
        ),
        ModelSpec(
            key="M6",
            display_name="M6 Best PIKA pre-finetune",
            family="best_pika_pre_finetune",
            checkpoint_filename="M6_best_pika_pre_ft.pth",
            artifact_subdir="M6_best_pika_pre_ft",
            needs_context=True,
            needs_graph=True,
            needs_prescription_branch=True,
            pill_backbone="tf_efficientnetv2_s.in21k_ft_in1k",
            prescription_backbone="resnet18.a1_in1k",
            notes="Best PIKA trước fine-tune.",
        ),
        ModelSpec(
            key="M7",
            display_name="M7 Fine-tune v1",
            family="best_pika_finetune_v1",
            checkpoint_filename="M7_finetune_v1_best.pth",
            artifact_subdir="M7_finetune_v1",
            needs_context=True,
            needs_graph=True,
            needs_prescription_branch=True,
            pill_backbone="tf_efficientnetv2_s.in21k_ft_in1k",
            prescription_backbone="resnet18.a1_in1k",
            notes="Fine-tune version 1.",
        ),
        ModelSpec(
            key="M8",
            display_name="M8 Fine-tune v2",
            family="best_pika_finetune_v2",
            checkpoint_filename="M8_finetune_v2_best.pth",
            artifact_subdir="M8_finetune_v2",
            needs_context=True,
            needs_graph=True,
            needs_prescription_branch=True,
            pill_backbone="tf_efficientnetv2_s.in21k_ft_in1k",
            prescription_backbone="resnet18.a1_in1k",
            notes="Fine-tune version 2.",
        ),
        ModelSpec(
            key="M9",
            display_name="M9 Fine-tune v3",
            family="best_pika_finetune_v3",
            checkpoint_filename="M9_finetune_v3_best.pth",
            artifact_subdir="M9_finetune_v3",
            needs_context=True,
            needs_graph=True,
            needs_prescription_branch=True,
            pill_backbone="tf_efficientnetv2_s.in21k_ft_in1k",
            prescription_backbone="resnet18.a1_in1k",
            notes="Fine-tune version 3 - bản mạnh nhất hiện tại.",
        ),
    ]

    registry = {}
    for spec in base_specs:
        resolved = resolve_model_spec(spec, model_dir=model_dir, artifact_root=artifact_root)
        registry[resolved.key] = resolved

    return registry


def validate_registry(registry: Dict[str, ModelSpec]) -> Dict[str, Dict[str, List[str]]]:
    """
    Trả về report thiếu file của từng model.
    """
    report = {}

    for key, spec in registry.items():
        missing = []
        existing = []

        for path in spec.expected_files():
            if os.path.exists(path):
                existing.append(path)
            else:
                missing.append(path)

        report[key] = {
            "existing": existing,
            "missing": missing,
        }

    return report


def print_registry_summary(registry: Dict[str, ModelSpec]) -> None:
    print("=" * 100)
    print("MODEL REGISTRY SUMMARY")
    print("=" * 100)

    for key, spec in registry.items():
        print(f"{key}: {spec.display_name}")
        print(f"  family               : {spec.family}")
        print(f"  checkpoint           : {spec.checkpoint_path}")
        print(f"  artifact_dir         : {spec.artifact_dir}")
        print(f"  needs_context        : {spec.needs_context}")
        print(f"  needs_graph          : {spec.needs_graph}")
        print(f"  needs_prescription   : {spec.needs_prescription_branch}")
        print(f"  pill_backbone        : {spec.pill_backbone}")
        print(f"  prescription_backbone: {spec.prescription_backbone}")
        print(f"  notes                : {spec.notes}")
        print("-" * 100)


def print_missing_report(registry: Dict[str, ModelSpec]) -> None:
    report = validate_registry(registry)

    print("=" * 100)
    print("MODEL ARTIFACT CHECK")
    print("=" * 100)

    for key, info in report.items():
        missing = info["missing"]
        existing = info["existing"]

        print(f"{key}:")
        print(f"  existing files: {len(existing)}")
        print(f"  missing files : {len(missing)}")

        if len(missing) > 0:
            for m in missing:
                print(f"    [MISSING] {m}")
        else:
            print("    [OK] tất cả file kỳ vọng đều đang có")
        print("-" * 100)


if __name__ == "__main__":
    # Bạn có thể sửa 2 path này khi test nhanh local/Colab
    model_dir = "./model"
    artifact_root = "./model_artifacts"

    registry = build_model_registry(model_dir=model_dir, artifact_root=artifact_root)
    print_registry_summary(registry)
    print_missing_report(registry)
