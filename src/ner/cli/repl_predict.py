"""交互式预测 REPL（只在需要时加载重依赖）

设计目标：
- 通过 `ner predict` 在缺省参数时启动交互式会话（REPL），仅提供 predict 能力（首期）。
- 首次加载模型时再懒加载 PyTorch/Transformers 等重依赖，后续多轮预测复用内存中的模型与分词器。
- 不影响原有命令行子命令（train/evaluate/...）；作为独立入口存在。

交互命令（首期最小集合）：
- load <model_path>       加载模型与分词器（仅加载一次，重复加载会覆盖）
- predict <text>          对单条文本进行预测（需要先 load）
- threshold <float>       设置预测置信度阈值（默认 0.5）
- format <json|text|conll>设置输出格式（默认 json）
- info                    显示当前会话状态
- help                    显示帮助
- exit/quit               退出 REPL

注意：
- 模块本身不在导入时加载任何重依赖，`load` 时才动态导入。
- 仅在需要的函数中进行 `from transformers import AutoTokenizer` 等导入。
"""

from __future__ import annotations

from typing import Optional


class PredictREPL:
    """预测 REPL，会话内可多轮使用同一模型，避免重复加载重依赖。

    参数：
    - logger: 可选的 logger 实例（建议复用 CLI 已有 logger）。
    """

    def __init__(self, logger=None) -> None:
        self.logger = logger
        self.model = None
        self.tokenizer = None
        self.model_path: Optional[str] = None
        self.output_format: str = "json"
        self.confidence_threshold: float = 0.5

    # -------------------------- 公共入口 --------------------------
    def run(self) -> None:
        """启动交互循环。

        说明：
        - 不会在启动时加载任何模型；仅响应用户的 `load` 指令时加载。
        - 使用简单的命令解析；未匹配到命令时，若已加载模型，将整行视为待预测文本。
        """
        self._print_banner()
        while True:
            try:
                line = input("NER-Predict> ").strip()
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
            elif cmd == "load":
                if not arg:
                    self._log_error("用法: load <model_path>")
                    continue
                self._handle_load(arg)
            elif cmd == "predict":
                if not arg:
                    self._log_error("用法: predict <text>")
                    continue
                self._handle_predict(arg)
            elif cmd == "threshold":
                self._handle_threshold(arg)
            elif cmd == "format":
                self._handle_format(arg)
            else:
                # 未知命令：如果已加载模型，则把整行当作文本预测；否则提示帮助
                if self.model is not None:
                    self._handle_predict(line)
                else:
                    self._log_error("未知命令。可先执行: load <model_path>；或输入 help 查看帮助。")

    # -------------------------- 命令处理 --------------------------
    def _handle_load(self, model_path: str) -> None:
        """加载模型与分词器（懒加载重依赖）。"""
        try:
            # 重依赖在此处加载，避免启动时的冷启动开销
            from ner.models import NERModelManager  # type: ignore
            from transformers import AutoTokenizer  # type: ignore

            manager = NERModelManager(logger=self.logger)
            model = manager.load_model(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

            self.model = model
            self.tokenizer = tokenizer
            self.model_path = model_path

            self._log_info(f"模型已加载: {model_path}")
        except Exception as e:
            self._log_error(f"加载模型失败: {e}")

    def _handle_predict(self, text: str) -> None:
        """对文本进行预测。"""
        if self.model is None or self.tokenizer is None:
            self._log_error("尚未加载模型。请先执行: load <model_path>")
            return

        try:
            prediction = self.model.predict(
                text,
                tokenizer=self.tokenizer,
                confidence_threshold=self.confidence_threshold,
            )

            if self.output_format == "json":
                import json
                print(json.dumps(prediction, ensure_ascii=False, indent=2))
            elif self.output_format == "text":
                tokens = prediction.get("tokens", [])
                labels = prediction.get("labels", [])
                for tok, lab in zip(tokens, labels):
                    print(f"{tok}\t{lab}")
            elif self.output_format == "conll":
                tokens = prediction.get("tokens", [])
                labels = prediction.get("labels", [])
                for tok, lab in zip(tokens, labels):
                    print(f"{tok} {lab}")
            else:
                self._log_error(f"未知输出格式: {self.output_format}")
        except Exception as e:
            self._log_error(f"预测失败: {e}")

    def _handle_threshold(self, arg: str) -> None:
        """设置预测置信度阈值。"""
        try:
            value = float(arg)
            if value < 0 or value > 1:
                raise ValueError("阈值需在 [0,1] 区间内")
            self.confidence_threshold = value
            self._log_info(f"已设置阈值: {self.confidence_threshold}")
        except Exception as e:
            self._log_error(f"阈值无效: {e}")

    def _handle_format(self, arg: str) -> None:
        """设置输出格式。支持 json/text/conll。"""
        fmt = (arg or "").strip().lower()
        if fmt in {"json", "text", "conll"}:
            self.output_format = fmt
            self._log_info(f"已设置输出格式: {self.output_format}")
        else:
            self._log_error("用法: format <json|text|conll>")

    # -------------------------- 辅助输出 --------------------------
    def _print_banner(self) -> None:
        print("NER 交互式预测会话 (REPL)")
        print("输入 help 查看帮助，exit/quit 退出。")

    def _print_help(self) -> None:
        print(
            """
可用命令：
  load <model_path>         加载模型与分词器（懒加载重库，仅首次慢）
  predict <text>            预测单条文本；已加载模型时，直接输入文本也可预测
  threshold <float>         设置预测置信度阈值（0~1，默认 0.5）
  format <json|text|conll>  设置输出格式（默认 json）
  info                      显示当前状态
  help                      显示帮助
  exit | quit               退出会话
            """.strip()
        )

    def _print_info(self) -> None:
        print(
            f"模型: {'已加载' if self.model is not None else '未加载'} | "
            f"路径: {self.model_path or '-'} | "
            f"阈值: {self.confidence_threshold} | 格式: {self.output_format}"
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


