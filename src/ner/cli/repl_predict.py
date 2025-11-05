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
        self.country: Optional[str] = None
        # text preprocessor (built from country config when设置)
        self.preprocessor = None

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
            elif cmd == "country":
                if not arg:
                    self._log_error("用法: country <code>")
                    continue
                self._handle_country(arg)
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

            if self.country:
                self._log_info(f"已设置国家: {self.country}，预处理已启用")
            else:
                self._log_info("尚未设置国家，预处理未启用")
        except Exception as e:
            self._log_error(f"加载模型失败: {e}")

    def _handle_country(self, country_code: str) -> None:
        """设置国家代码并从配置构建预处理器。"""
        try:
            from ner.preprocess import build_preprocessor_from_config  # type: ignore
            from ner.config import ConfigManager  # type: ignore

            manager = ConfigManager()
            config = manager.load_country_config(country_code)
            self.preprocessor = build_preprocessor_from_config(config)
            self._log_info(f'加载国家配置成功，预处理已启用：{self.preprocessor}')
            self.country = country_code
            self._log_info(f"已设置国家: {country_code}，预处理已启用")
        except Exception as e:
            self._log_error(f"设置国家失败: {e}")

    def _handle_predict(self, text: str) -> None:
        """对文本进行预测。"""
        if self.model is None or self.tokenizer is None:
            self._log_error("尚未加载模型。请先执行: load <model_path>")
            return

        try:
            input_text = text
            if getattr(self, 'preprocessor', None):
                input_text = self.preprocessor.apply_text(input_text)
            prediction = self.model.predict(
                input_text,
                tokenizer=self.tokenizer,
                confidence_threshold=self.confidence_threshold,
            )

            if self.output_format == "json":
                import json
                print(json.dumps(prediction, ensure_ascii=False, indent=2))
            elif self.output_format == "text":
                tokens = prediction.get("tokens", [])
                labels = prediction.get("labels", [])
                confidences = prediction.get("confidences", [])
                # 计算各列的最大宽度以对齐输出
                max_token_len = max((len(str(tok)) for tok in tokens), default=0)
                max_label_len = max((len(str(lab)) for lab in labels), default=0)
                # 确保最小宽度，并添加适当间距
                token_width = max(max_token_len, 10)
                label_width = max(max_label_len, 15)
                # 打印表头
                print(f"{'Token':<{token_width}}  {'Label':<{label_width}}  Confidence")
                print("-" * (token_width + label_width + 25))
                # 打印对齐的数据行
                for tok, lab, conf in zip(tokens, labels, confidences):
                    print(f"{tok:<{token_width}}  {lab:<{label_width}}  {conf:.4f}")
            elif self.output_format == "conll":
                tokens = prediction.get("tokens", [])
                labels = prediction.get("labels", [])
                confidences = prediction.get("confidences", [])
                for tok, lab, conf in zip(tokens, labels, confidences):
                    print(f"{tok} {lab} {conf:.4f}")
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
  country <code>            设置国家代码（例如：uae）
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
            f"国家: {self.country or '-'} | "
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


