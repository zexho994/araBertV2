"""
Hugging Face Hub uploader utility

功能:
- 从 `data/ner/models/` 下选择/定位模型目录
- 生成 Hugging Face 兼容的必要文件: README.md, .gitattributes, labels.json
- 可选复制分词器文件（若存在）
- 通过 huggingface_hub 一键上传整个目录

用法示例:
  1) 仅准备目录(不会上传):
     python -m src.utils.hf_uploader prepare --model-path data/ner/models/uae/uae_model_20250101_120000

  2) 直接上传(自动准备，若未准备):
     python -m src.utils.hf_uploader upload \
       --model-path data/ner/models/uae/uae_model_20250101_120000 \
       --repo-id your-username/uae-ner-bert \
       --private

  3) 通过注册表选择最新模型并上传:
     python -m src.utils.hf_uploader upload-latest \
       --country uae \
       --repo-id your-username/uae-ner-latest

认证:
- 推荐使用环境变量 HF_TOKEN, 或已在本机登录 `huggingface-cli login`
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder, whoami

# 项目内导入(用于 registry 定位与日志)
try:
    from src.ner.models import NERModelManager  # type: ignore
except Exception:
    # 允许在不同入口运行; 尝试相对导入
    try:
        from ner.models import NERModelManager  # type: ignore
    except Exception:
        NERModelManager = None  # lazy path-only mode


HUB_README_TEMPLATE = """---
language: [ar]
license: apache-2.0
tags:
- token-classification
datasets:
- unknown
library_name: transformers
pipeline_tag: token-classification
---

# {title}

本仓库存放一个命名实体识别(NER)模型的权重与配置。

- 任务: Token Classification (NER)
- 国家/区域: {country}
- 标签数: {num_labels}

## 使用方式

加载到 transformers 中进行推理：

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_id = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForTokenClassification.from_pretrained(model_id, ignore_mismatched_sizes=True)

text = "مثال على العنوان"
inputs = tokenizer(text.split(), is_split_into_words=True, return_tensors="pt", truncation=True)
outputs = model(**inputs)
```

注意: 如果该模型来自自定义训练代码，参数名可能与标准 `BertForTokenClassification` 对齐；若出现不匹配，可设置 `ignore_mismatched_sizes=True` 或自行适配。

## 标签映射

`labels.json` 包含 `id2label` 与 `label2id` 用于解释预测结果。

"""


def _read_label_mappings(model_dir: Path) -> Tuple[Dict[str, int], Dict[str, str]]:
    """从模型目录提取 label2id / id2label。

    支持两种来源:
    1) 管理器保存的自定义 config.json: { label2id, id2label }
    2) transformers 保存的 config.json: 同上字段
    若无法读取，回退为空映射。
    """
    label2id: Dict[str, int] = {}
    id2label: Dict[str, str] = {}

    cfg = model_dir / "config.json"
    if cfg.exists():
        try:
            with open(cfg, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 兼容两类结构
            if isinstance(data, dict):
                if "label2id" in data and "id2label" in data:
                    label2id = {str(k): int(v) for k, v in data.get("label2id", {}).items()}
                    # id2label 可能键为字符串或数字
                    id2label = {str(k): str(v) for k, v in data.get("id2label", {}).items()}
                elif "config" in data and isinstance(data["config"], dict):
                    inner = data["config"]
                    if "label2id" in inner and "id2label" in inner:
                        label2id = {str(k): int(v) for k, v in inner.get("label2id", {}).items()}
                        id2label = {str(k): str(v) for k, v in inner.get("id2label", {}).items()}
        except Exception:
            pass

    # 若仍为空，尝试从 labels.json
    if not label2id and (model_dir / "labels.json").exists():
        try:
            with open(model_dir / "labels.json", "r", encoding="utf-8") as f:
                d = json.load(f)
            label2id = {str(k): int(v) for k, v in d.get("label2id", {}).items()}
            id2label = {str(k): str(v) for k, v in d.get("id2label", {}).items()}
        except Exception:
            pass

    return label2id, id2label


def _write_labels_json(model_dir: Path, label2id: Dict[str, int], id2label: Dict[str, str]) -> None:
    payload = {"label2id": label2id, "id2label": id2label}
    with open(model_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _ensure_gitattributes(model_dir: Path) -> None:
    content = (
        "*.bin filter=lfs diff=lfs merge=lfs -text\n"
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
    )
    path = model_dir / ".gitattributes"
    if not path.exists():
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


def _maybe_copy_tokenizer_files(src_dir: Path, dst_dir: Path) -> None:
    """Copy tokenizer-related files if needed.

    Safe on Windows: skips when src == dst or destination already exists.
    """
    candidates = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "merges.txt",
        "special_tokens_map.json",
        "spiece.model",
    ]
    for name in candidates:
        src = src_dir / name
        dst = dst_dir / name
        if not src.exists():
            continue
        try:
            # Avoid copying onto itself or overwriting existing files
            if dst.exists():
                continue
            if src.resolve() == dst.resolve():
                continue
        except Exception:
            # Fallback path equality check if resolve() fails
            if os.path.abspath(str(src)) == os.path.abspath(str(dst)):
                continue
        try:
            shutil.copy2(src, dst)
        except PermissionError:
            # File may be locked on Windows; skip gracefully
            pass


def _write_readme(model_dir: Path, repo_id: str, country: str, num_labels: int, title: Optional[str] = None) -> None:
    title = title or f"NER model ({country})"
    text = HUB_README_TEMPLATE.format(title=title, country=country, num_labels=num_labels, repo_id=repo_id)
    with open(model_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(text)


def prepare_model_dir(model_path: str, repo_id: str = "", country: str = "unknown") -> Dict[str, Any]:
    """生成必要文件，使目录更适配 Hugging Face Hub.

    返回字典包含: model_dir, num_labels, label2id, id2label
    """
    model_dir = Path(model_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # 权重校验
    if not (model_dir / "pytorch_model.bin").exists() and not (model_dir / "model.safetensors").exists():
        raise FileNotFoundError("Weights not found (pytorch_model.bin or model.safetensors)")

    # 标签映射
    label2id, id2label = _read_label_mappings(model_dir)
    num_labels = len(label2id) if label2id else len(id2label)

    # 写出 labels.json 以便下游读取
    if label2id or id2label:
        _write_labels_json(model_dir, label2id, id2label)

    # .gitattributes for LFS
    _ensure_gitattributes(model_dir)

    # 复制分词器文件(若存在于本目录或上级目录)
    _maybe_copy_tokenizer_files(model_dir, model_dir)

    # README
    if repo_id:
        _write_readme(model_dir, repo_id=repo_id, country=country, num_labels=num_labels or 0)

    return {
        "model_dir": str(model_dir),
        "num_labels": num_labels,
        "label2id": label2id,
        "id2label": id2label,
    }


def _resolve_model_path_by_registry(country: str, model_root: str = "data/ner/models") -> str:
    if NERModelManager is None:
        raise RuntimeError("NERModelManager is not available; please pass --model-path explicitly.")
    mgr = NERModelManager(model_dir=model_root)
    latest_id = mgr.get_latest_model(country)
    if latest_id is None:
        raise ValueError(f"No active models found for country: {country}")
    info = mgr.get_model_info(latest_id)
    return info.get("path")


def upload_to_hub(model_path: str, repo_id: str, token: Optional[str] = None, private: bool = False, revision: Optional[str] = None) -> str:
    """将目录上传到 Hugging Face Hub。若仓库不存在则自动创建。
    返回仓库 web url。
    """
    model_dir = Path(model_path)
    if token:
        HfFolder.save_token(token)

    api = HfApi()

    # 创建或确认仓库
    try:
        create_repo(repo_id, private=private, exist_ok=True, token=token)
    except Exception:
        pass

    # 上传整个目录
    upload_folder(
        repo_id=repo_id,
        folder_path=str(model_dir),
        path_in_repo=".",
        commit_message="Upload NER model",
        token=token,
        revision=revision,
        ignore_patterns=["**/__pycache__/**", "**/*.pt", "**/*.ckpt"],
    )

    # 返回 URL
    user = whoami(token=token)
    owner = repo_id.split("/")[0] if "/" in repo_id else user.get("name", "")
    name = repo_id.split("/")[-1]
    return f"https://huggingface.co/{owner}/{name}"


def main():
    parser = argparse.ArgumentParser(description="Prepare and upload NER models to Hugging Face Hub")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # prepare
    p_prepare = sub.add_parser("prepare", help="Prepare a model directory for Hub")
    p_prepare.add_argument("--model-path", required=False, help="Path to model directory")
    p_prepare.add_argument("--country", default="unknown", help="Country code to annotate")
    p_prepare.add_argument("--repo-id", default="", help="Optional repo id for generating README")
    p_prepare.add_argument("--use-registry", action="store_true", help="Resolve model path via registry and country")

    # upload
    p_upload = sub.add_parser("upload", help="Upload a model directory to Hub")
    p_upload.add_argument("--model-path", required=False, help="Path to model directory")
    p_upload.add_argument("--repo-id", required=True, help="Target repo id, e.g., username/repo")
    p_upload.add_argument("--country", default="unknown", help="Country code for README")
    p_upload.add_argument("--token", default=os.environ.get("HF_TOKEN", None), help="HF token or set HF_TOKEN env")
    p_upload.add_argument("--private", action="store_true", help="Create private repo")
    p_upload.add_argument("--use-registry", action="store_true", help="Resolve model path via registry and country")

    # upload-latest by country
    p_latest = sub.add_parser("upload-latest", help="Upload latest model for a country via registry")
    p_latest.add_argument("--country", required=True, help="Country code")
    p_latest.add_argument("--repo-id", required=True, help="Target repo id")
    p_latest.add_argument("--token", default=os.environ.get("HF_TOKEN", None), help="HF token or set HF_TOKEN env")
    p_latest.add_argument("--private", action="store_true", help="Create private repo")
    p_latest.add_argument("--model-root", default="data/ner/models", help="Model root directory")

    args = parser.parse_args()

    if args.cmd == "prepare":
        if args.use_registry:
            if not args.country:
                raise SystemExit("--country is required with --use-registry")
            model_path = _resolve_model_path_by_registry(args.country)
        else:
            if not args.model_path:
                raise SystemExit("--model-path is required when not using registry")
            model_path = args.model_path

        info = prepare_model_dir(model_path, repo_id=args.repo_id, country=args.country)
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return

    if args.cmd == "upload":
        if args.use_registry:
            if not args.country:
                raise SystemExit("--country is required with --use-registry")
            model_path = _resolve_model_path_by_registry(args.country)
        else:
            if not args.model_path:
                raise SystemExit("--model-path is required when not using registry")
            model_path = args.model_path

        # prepare & upload
        prepare_model_dir(model_path, repo_id=args.repo_id, country=args.country)
        url = upload_to_hub(model_path, repo_id=args.repo_id, token=args.token, private=args.private)
        print(f"Uploaded to: {url}")
        return

    if args.cmd == "upload-latest":
        model_path = _resolve_model_path_by_registry(args.country, model_root=args.model_root)
        prepare_model_dir(model_path, repo_id=args.repo_id, country=args.country)
        url = upload_to_hub(model_path, repo_id=args.repo_id, token=args.token, private=args.private)
        print(f"Uploaded to: {url}")
        return


if __name__ == "__main__":
    main()


