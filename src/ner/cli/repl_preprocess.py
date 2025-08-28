"""交互式预处理 REPL

目标：
- 通过 `ner preprocess` 在缺省参数时启动交互式会话，仅提供文本预处理能力。
- 用户先设置国家 `country <code>` 加载该国家的 preprocess pipeline，然后输入文本进行处理。
- 懒加载：仅在需要时导入配置与预处理模块。
"""

from __future__ import annotations

from typing import Optional


class PreprocessREPL:
    """预处理 REPL：会话内可多次复用同一预处理管道。

    命令：
    - country <code>   加载并启用对应国家的预处理管道
    - apply <text>     对文本执行预处理（已设置国家时生效）
    - csv <in>         读取CSV的formatted_address列，生成preprocessed_address并写回原文件
    - info             显示当前状态
    - help             显示帮助
    - exit/quit        退出会话
    """

    def __init__(self, logger=None) -> None:
        self.logger = logger
        self.country: Optional[str] = None
        self.preprocessor = None

    # -------------------------- 入口 --------------------------
    def run(self) -> None:
        self._print_banner()
        while True:
            try:
                line = input("NER-Preprocess> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("")
                self._log_info("Bye.")
                break

            if not line:
                continue

            cmd, *args = line.split(maxsplit=1)
            arg = args[0] if args else ""

            if cmd in {"exit", "quit"}:
                self._log_info("Bye.")
                break
            elif cmd == "help":
                self._print_help()
            elif cmd == "info":
                self._print_info()
            elif cmd == "country":
                if not arg:
                    self._log_error("用法: country <code>")
                    continue
                self._handle_country(arg)
            elif cmd == "apply":
                if not arg:
                    self._log_error("用法: apply <text>")
                    continue
                self._handle_apply(arg)
            elif cmd == "csv":
                if not arg:
                    self._log_error("用法: csv <input_csv>")
                    continue
                self._handle_csv(arg)
            else:
                # 若已加载预处理器，则将整行作为文本处理
                if self.preprocessor is not None:
                    self._handle_apply(line)
                else:
                    self._log_error("未知命令。请先执行: country <code>；或输入 help 查看帮助。")

    # -------------------------- 命令处理 --------------------------
    def _handle_country(self, country_code: str) -> None:
        try:
            from ner.config import ConfigManager  # type: ignore
            from ner.preprocess import build_preprocessor_from_config  # type: ignore

            manager = ConfigManager()
            config = manager.load_country_config(country_code)
            self.preprocessor = build_preprocessor_from_config(config)
            self.country = country_code
            self._log_info(f"已设置国家: {country_code}，预处理管道已就绪。")
        except Exception as e:
            self._log_error(f"设置国家失败: {e}")

    def _handle_apply(self, text: str) -> None:
        if self.preprocessor is None:
            self._log_error("尚未设置国家。请先执行: country <code>")
            return
        try:
            output = self.preprocessor.apply_text(text)
            print(output)
        except Exception as e:
            self._log_error(f"预处理失败: {e}")

    def _handle_csv(self, arg: str) -> None:
        if self.preprocessor is None:
            self._log_error("尚未设置国家。请先执行: country <code>")
            return
        try:
            import os
            import pandas as pd  # type: ignore
        except Exception as e:
            self._log_error(f"依赖未就绪，请确保已安装 pandas。错误: {e}")
            return

        parts = [p for p in arg.split() if p]
        if len(parts) < 1:
            self._log_error("用法: csv <input_csv>")
            return

        input_csv = parts[0]
        if not os.path.isfile(input_csv):
            self._log_error(f"文件不存在: {input_csv}")
            return

        # 按需求：直接回写到输入文件
        output_csv = input_csv

        try:
            df = pd.read_csv(input_csv)
        except Exception as e:
            self._log_error(f"读取CSV失败: {e}")
            return

        if "formatted_address" not in df.columns:
            self._log_error("CSV缺少列: formatted_address")
            return

        try:
            def _safe_process(val):
                if pd.isna(val):
                    return ""
                try:
                    return self.preprocessor.apply_text(str(val))
                except Exception:
                    return ""

            df["preprocessed_address"] = df["formatted_address"].apply(_safe_process)
        except Exception as e:
            self._log_error(f"批量预处理失败: {e}")
            return

        try:
            df.to_csv(output_csv, index=False)
            self._log_info(f"处理完成，已保存至: {output_csv}")
        except Exception as e:
            self._log_error(f"保存CSV失败: {e}")

    # -------------------------- 辅助输出 --------------------------
    def _print_banner(self) -> None:
        print("NER 交互式预处理会话 (REPL)")
        print("输入 help 查看帮助，exit/quit 退出。")

    def _print_help(self) -> None:
        print(
            """
可用命令：
  country <code>   加载并启用对应国家的预处理管道
  apply <text>     对文本执行预处理；已加载管道时直接输入文本也可处理
  csv <in>         读取CSV formatted_address，生成 preprocessed_address 并写回原文件
  info             显示当前状态
  help             显示帮助
  exit | quit      退出会话
            """.strip()
        )

    def _print_info(self) -> None:
        print(
            f"国家: {self.country or '-'} | 预处理: {'已就绪' if self.preprocessor is not None else '未加载'}"
        )

    def _log_info(self, message: str) -> None:
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def _log_error(self, message: str) -> None:
        if self.logger:
            self.logger.error(message)
        else:
            print(f"ERROR: {message}")


