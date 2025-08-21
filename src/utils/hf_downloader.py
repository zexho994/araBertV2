"""
Hugging Face Hub 模型下载工具

功能:
- 从 Hugging Face Hub 下载指定模型到本地路径
- 支持下载完整模型目录或特定文件
- 提供进度显示和错误处理
- 支持私有仓库（需要token认证）
- 可配置缓存和重新下载选项

用法示例:
  1) 下载完整模型:
     python -m src.utils.hf_downloader download \
       --repo-id microsoft/DialoGPT-medium \
       --local-dir ./models/DialoGPT-medium

  2) 下载特定文件:
     python -m src.utils.hf_downloader download-file \
       --repo-id microsoft/DialoGPT-medium \
       --filename pytorch_model.bin \
       --local-dir ./models/DialoGPT-medium

  3) 列出仓库文件:
     python -m src.utils.hf_downloader list-files \
       --repo-id microsoft/DialoGPT-medium

认证:
- 使用环境变量 HF_TOKEN 或通过 huggingface-cli login 登录
- 对于私有仓库，必须提供有效的访问token
"""

import os
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
from tqdm import tqdm

try:
    from huggingface_hub import (
        HfApi, 
        hf_hub_download, 
        snapshot_download,
        list_repo_files,
        HfFolder,
        login,
        whoami
    )
    from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
except ImportError:
    raise ImportError(
        "huggingface_hub is required. Install it with: pip install huggingface_hub"
    )


class HFDownloader:
    """Hugging Face Hub 下载器"""
    
    def __init__(self, token: Optional[str] = None):
        """
        初始化下载器
        
        Args:
            token: HF访问token，如果为None则从环境变量或本地配置读取
        """
        self.token = token or os.environ.get("HF_TOKEN") or HfFolder.get_token()
        self.api = HfApi(token=self.token)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def verify_token(self) -> bool:
        """验证token是否有效"""
        try:
            if self.token:
                user_info = whoami(token=self.token)
                self.logger.info(f"认证成功，用户: {user_info.get('name', 'unknown')}")
                return True
            else:
                self.logger.warning("未提供HF token，只能访问公开仓库")
                return False
        except Exception as e:
            self.logger.error(f"Token验证失败: {e}")
            return False
    
    def check_repo_exists(self, repo_id: str, repo_type: str = "model") -> bool:
        """检查仓库是否存在"""
        try:
            self.api.repo_info(repo_id=repo_id, repo_type=repo_type, token=self.token)
            return True
        except (RepositoryNotFoundError, HfHubHTTPError):
            return False
        except Exception as e:
            self.logger.error(f"检查仓库时出错: {e}")
            return False
    
    def list_repo_files(self, repo_id: str, repo_type: str = "model") -> List[str]:
        """列出仓库中的所有文件"""
        try:
            files = list_repo_files(
                repo_id=repo_id,
                repo_type=repo_type,
                token=self.token
            )
            return sorted(files)
        except Exception as e:
            self.logger.error(f"获取文件列表失败: {e}")
            return []
    
    def download_file(
        self,
        repo_id: str,
        filename: str,
        local_dir: str,
        repo_type: str = "model",
        revision: Optional[str] = None,
        force_download: bool = False
    ) -> Optional[str]:
        """
        下载单个文件
        
        Args:
            repo_id: 仓库ID，格式如 "microsoft/DialoGPT-medium"
            filename: 要下载的文件名
            local_dir: 本地保存目录
            repo_type: 仓库类型 ("model", "dataset", "space")
            revision: 版本/分支，默认为main
            force_download: 是否强制重新下载
            
        Returns:
            下载文件的本地路径，失败时返回None
        """
        try:
            # 确保目录存在
            os.makedirs(local_dir, exist_ok=True)
            
            self.logger.info(f"开始下载文件: {repo_id}/{filename}")
            
            # 下载文件
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=local_dir,
                repo_type=repo_type,
                revision=revision,
                token=self.token,
                force_download=force_download,
                local_files_only=False
            )
            
            self.logger.info(f"文件下载完成: {downloaded_path}")
            return downloaded_path
            
        except Exception as e:
            self.logger.error(f"下载文件失败: {e}")
            return None
    
    def download_model(
        self,
        repo_id: str,
        local_dir: str,
        repo_type: str = "model",
        revision: Optional[str] = None,
        ignore_patterns: Optional[List[str]] = None,
        force_download: bool = False,
        max_workers: int = 8
    ) -> Optional[str]:
        """
        下载完整模型/数据集
        
        Args:
            repo_id: 仓库ID
            local_dir: 本地保存目录
            repo_type: 仓库类型
            revision: 版本/分支
            ignore_patterns: 忽略的文件模式列表
            force_download: 是否强制重新下载
            max_workers: 并行下载的最大工作线程数
            
        Returns:
            下载目录路径，失败时返回None
        """
        try:
            # 检查仓库是否存在
            if not self.check_repo_exists(repo_id, repo_type):
                self.logger.error(f"仓库不存在或无访问权限: {repo_id}")
                return None
            
            # 确保目录存在
            os.makedirs(local_dir, exist_ok=True)
            
            self.logger.info(f"开始下载模型: {repo_id} -> {local_dir}")
            
            # 默认忽略模式
            if ignore_patterns is None:
                ignore_patterns = [
                    "*.git*",
                    "README.md",
                    "*.md",
                    ".gitattributes"
                ]
            
            # 下载完整仓库
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=None,  # 不使用缓存，直接下载到指定目录
                local_dir=local_dir,
                repo_type=repo_type,
                revision=revision,
                token=self.token,
                ignore_patterns=ignore_patterns,
                force_download=force_download,
                max_workers=max_workers,
                local_files_only=False
            )
            
            self.logger.info(f"模型下载完成: {downloaded_path}")
            return downloaded_path
            
        except Exception as e:
            self.logger.error(f"下载模型失败: {e}")
            return None
    
    def get_model_info(self, repo_id: str, repo_type: str = "model") -> Optional[Dict[str, Any]]:
        """获取模型信息"""
        try:
            info = self.api.model_info(repo_id=repo_id, token=self.token)
            return {
                "id": info.modelId,
                "author": info.author,
                "downloads": info.downloads,
                "likes": info.likes,
                "tags": info.tags,
                "pipeline_tag": info.pipeline_tag,
                "library_name": info.library_name,
                "created_at": str(info.created_at) if info.created_at else None,
                "last_modified": str(info.last_modified) if info.last_modified else None,
            }
        except Exception as e:
            self.logger.error(f"获取模型信息失败: {e}")
            return None
    
    def download_with_progress(
        self,
        repo_id: str,
        local_dir: str,
        files_to_download: Optional[List[str]] = None,
        **kwargs
    ) -> bool:
        """
        带进度显示的下载
        
        Args:
            repo_id: 仓库ID
            local_dir: 本地目录
            files_to_download: 指定要下载的文件列表，None表示下载全部
            **kwargs: 其他下载参数
            
        Returns:
            是否下载成功
        """
        try:
            if files_to_download:
                # 下载指定文件
                success_count = 0
                with tqdm(total=len(files_to_download), desc="下载文件") as pbar:
                    for filename in files_to_download:
                        result = self.download_file(
                            repo_id=repo_id,
                            filename=filename,
                            local_dir=local_dir,
                            **kwargs
                        )
                        if result:
                            success_count += 1
                        pbar.update(1)
                        pbar.set_postfix({"成功": success_count, "失败": len(files_to_download) - success_count})
                
                return success_count == len(files_to_download)
            else:
                # 下载完整模型
                result = self.download_model(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    **kwargs
                )
                return result is not None
                
        except Exception as e:
            self.logger.error(f"下载过程中出错: {e}")
            return False


def main():
    """命令行接口
    python -m src.utils.hf_downloader download \
       --repo-id zexho/uae_address_roberta_v1.0 \
       --local-dir data/ner/models/uae_address_roberta_v1.0
    """
    parser = argparse.ArgumentParser(description="Hugging Face Hub 模型下载工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 通用参数
    def add_common_args(parser):
        parser.add_argument("--repo-id", required=True, help="仓库ID (e.g., microsoft/DialoGPT-medium)")
        parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF访问token")
        parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"], help="仓库类型")
        parser.add_argument("--revision", help="版本/分支名")
    
    # download 命令
    download_parser = subparsers.add_parser("download", help="下载完整模型")
    add_common_args(download_parser)
    download_parser.add_argument("--local-dir", required=True, help="本地保存目录")
    download_parser.add_argument("--force", action="store_true", help="强制重新下载")
    download_parser.add_argument("--ignore-patterns", nargs="*", help="忽略的文件模式")
    download_parser.add_argument("--max-workers", type=int, default=8, help="并行下载线程数")
    
    # download-file 命令
    file_parser = subparsers.add_parser("download-file", help="下载单个文件")
    add_common_args(file_parser)
    file_parser.add_argument("--filename", required=True, help="要下载的文件名")
    file_parser.add_argument("--local-dir", required=True, help="本地保存目录")
    file_parser.add_argument("--force", action="store_true", help="强制重新下载")
    
    # list-files 命令
    list_parser = subparsers.add_parser("list-files", help="列出仓库文件")
    add_common_args(list_parser)
    
    # info 命令
    info_parser = subparsers.add_parser("info", help="获取模型信息")
    add_common_args(info_parser)
    
    # verify-token 命令
    token_parser = subparsers.add_parser("verify-token", help="验证token")
    token_parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF访问token")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 创建下载器
    downloader = HFDownloader(token=getattr(args, 'token', None))
    
    if args.command == "verify-token":
        success = downloader.verify_token()
        exit(0 if success else 1)
    
    if args.command == "download":
        success = downloader.download_model(
            repo_id=args.repo_id,
            local_dir=args.local_dir,
            repo_type=args.repo_type,
            revision=args.revision,
            ignore_patterns=args.ignore_patterns,
            force_download=args.force,
            max_workers=args.max_workers
        )
        if success:
            print(f"✅ 模型下载成功: {args.local_dir}")
        else:
            print("❌ 模型下载失败")
            exit(1)
    
    elif args.command == "download-file":
        result = downloader.download_file(
            repo_id=args.repo_id,
            filename=args.filename,
            local_dir=args.local_dir,
            repo_type=args.repo_type,
            revision=args.revision,
            force_download=args.force
        )
        if result:
            print(f"✅ 文件下载成功: {result}")
        else:
            print("❌ 文件下载失败")
            exit(1)
    
    elif args.command == "list-files":
        files = downloader.list_repo_files(
            repo_id=args.repo_id,
            repo_type=args.repo_type
        )
        if files:
            print(f"📁 仓库 {args.repo_id} 包含 {len(files)} 个文件:")
            for file in files:
                print(f"  - {file}")
        else:
            print("❌ 获取文件列表失败")
            exit(1)
    
    elif args.command == "info":
        info = downloader.get_model_info(
            repo_id=args.repo_id,
            repo_type=args.repo_type
        )
        if info:
            print(f"📊 模型信息: {args.repo_id}")
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print("❌ 获取模型信息失败")
            exit(1)


if __name__ == "__main__":
    main()
