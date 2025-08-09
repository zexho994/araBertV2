"""CSV Annotation Generator

Reads a CSV file with a formatted address column and entity columns (lowercased
as defined in the country config), then generates BIO-tagged annotations per row
and saves them under the country-specific data directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import json
import re
import sys

import pandas as pd

# Ensure running this file directly works by adding the project 'src' to sys.path
# so that 'import ner' resolves to 'src/ner'
sys.path.append(str(Path(__file__).resolve().parents[2]))

from ner import DEFAULT_DATA_DIR
from ner.config.manager import ConfigManager
from ner.utils.text_utils import TextUtils


@dataclass
class CSVAnnotationGeneratorConfig:
    csv_path: str
    country_code: str
    output_file: Optional[str] = None  # If None, defaults to data dir / country / generated.json
    text_column: str = "formatted_address"


class CSVAnnotationGenerator:
    """Generate NER annotations from a CSV using country-specific labels.

    CSV requirements:
    - Must contain a text column (default: "formatted_address")
    - Must contain one or more entity columns whose names are the lowercase
      forms of the entity labels configured for the country (without BIO prefixes)
      e.g., for entity CITY -> column name "city".

    Output format:
    - Defaults to JSON Lines (JSONL), one example per line, saved under
      data/ner/data/<country_code>/generated.jsonl
    - If output_file ends with .json, a JSON array will be written instead.
    """

    def __init__(self, config_dir: str = "data/ner/configs") -> None:
        self.config_manager = ConfigManager(config_dir=config_dir)

    def generate(self, cfg: CSVAnnotationGeneratorConfig) -> Path:
        # Load country config and derive entity set
        country_cfg = self.config_manager.load_country_config(cfg.country_code)
        entity_names = self._derive_entity_names_from_config(country_cfg)
        entity_to_column = {e: e.lower() for e in entity_names}

        # Read CSV with delimiter auto-detection and clean headers
        df = pd.read_csv(cfg.csv_path, sep=None, engine="python")
        # Remove BOM and surrounding whitespace from headers
        df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
        # Build normalized header map
        norm_to_actual = {self._normalize_colname(c): c for c in df.columns}

        # Validate columns
        norm_text_col = self._normalize_colname(cfg.text_column)
        if norm_text_col not in norm_to_actual:
            raise ValueError(
                f"CSV missing required text column '{cfg.text_column}'. Columns: {list(df.columns)}"
            )
        text_col_actual = norm_to_actual[norm_text_col]

        # Map expected entity lower names to actual present column names via normalized match
        entity_to_actual_col: Dict[str, str] = {}
        for entity, expected_col in entity_to_column.items():
            norm_expected = self._normalize_colname(expected_col)
            if norm_expected in norm_to_actual:
                entity_to_actual_col[entity] = norm_to_actual[norm_expected]

        present_entity_columns = list(entity_to_actual_col.values())
        if not present_entity_columns:
            raise ValueError(
                "CSV does not contain any expected entity columns. "
                f"Expected any of: {sorted(entity_to_column.values())}. Columns: {list(df.columns)}"
            )

        # Build annotations
        examples: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            text = str(row.get(text_col_actual, "")).strip()
            if not text:
                continue

            tokens, token_spans = self._whitespace_tokenize_with_spans(text)
            entities = self._extract_entities_from_row(
                row=row,
                entity_to_column=entity_to_actual_col,
                tokens=tokens,
                token_spans=token_spans,
                original_text=text,
            )

            labels = TextUtils.convert_entities_to_bio(tokens, entities)
            examples.append({
                "text": text,
                "tokens": tokens,
                "labels": labels,
            })

        # Determine output path (default to JSONL)
        default_dir = Path("data/ner/data") / cfg.country_code
        output_path = Path(cfg.output_file) if cfg.output_file else default_dir / "generated.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write according to extension
        suffix = output_path.suffix.lower()
        if suffix == ".jsonl":
            with output_path.open("w", encoding="utf-8") as f:
                for ex in examples:
                    f.write(json.dumps(ex, ensure_ascii=False))
                    f.write("\n")
        elif suffix == ".json":
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(examples, f, ensure_ascii=False, indent=2)
        else:
            # Fallback to JSONL if unknown extension
            with (output_path.with_suffix(".jsonl")).open("w", encoding="utf-8") as f:
                for ex in examples:
                    f.write(json.dumps(ex, ensure_ascii=False))
                    f.write("\n")

        return output_path

    # Convenience helper for direct invocation without constructing config objects
    def generate_from_csv(self, csv_path: str, country_code: str, *, output_file: Optional[str] = None,
                          text_column: str = "formatted_address") -> Path:
        cfg = CSVAnnotationGeneratorConfig(
            csv_path=csv_path,
            country_code=country_code,
            output_file=output_file,
            text_column=text_column,
        )
        return self.generate(cfg)

    @staticmethod
    def _derive_entity_names_from_config(country_cfg: Dict[str, Any]) -> List[str]:
        """Derive entity names from config labels section.

        Supports two shapes:
        - labels.entities = ["CITY", "STREET", ...]
        - labels.label_names = ["O", "B-CITY", "I-CITY", ...]
        """
        labels_cfg = country_cfg.get("labels", {})

        entities: List[str] = []
        if isinstance(labels_cfg.get("entities"), list) and labels_cfg["entities"]:
            entities = [str(e).upper() for e in labels_cfg["entities"]]
        elif isinstance(labels_cfg.get("label_names"), list) and labels_cfg["label_names"]:
            uniq: Dict[str, None] = {}
            for name in labels_cfg["label_names"]:
                if not isinstance(name, str):
                    continue
                if name == "O":
                    continue
                if name.startswith("B-") or name.startswith("I-"):
                    ent = name[2:].upper()
                else:
                    ent = name.upper()
                uniq[ent] = None
            entities = list(uniq.keys())
        else:
            raise ValueError("Could not derive entity names from config.labels")

        return entities

    @staticmethod
    def _whitespace_tokenize_with_spans(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Tokenize on whitespace and return tokens and (start,end) char spans for each token."""
        tokens: List[str] = []
        spans: List[Tuple[int, int]] = []
        for m in re.finditer(r"\S+", text):
            token = m.group(0)
            tokens.append(token)
            spans.append((m.start(), m.end()))
        return tokens, spans

    @staticmethod
    def _find_char_span(text: str, value: str) -> Optional[Tuple[int, int]]:
        """Return the first (start,end) char span of value in text, or None if not found."""
        if not isinstance(value, str):
            return None
        s = value.strip()
        if not s:
            return None
        idx = text.find(s)
        if idx == -1:
            return None
        return idx, idx + len(s)

    @staticmethod
    def _normalize_colname(name: str) -> str:
        """Normalize a column name for matching: strip, remove BOM, lowercase, unify separators."""
        if name is None:
            return ""
        s = str(name).lstrip("\ufeff").strip().lower()
        # Replace whitespace and hyphens with underscores
        s = re.sub(r"[\s\-]+", "_", s)
        # Collapse multiple underscores
        s = re.sub(r"_+", "_", s)
        return s

    def _extract_entities_from_row(
        self,
        row: pd.Series,
        entity_to_column: Dict[str, str],
        tokens: List[str],
        token_spans: List[Tuple[int, int]],
        original_text: str,
    ) -> List[Dict[str, Any]]:
        """Extract entities as token-span dicts: {type,start,end} (end exclusive)."""
        text_values: Dict[str, str] = {}
        for entity, col in entity_to_column.items():
            if col in row and pd.notna(row[col]):
                val = str(row[col]).strip()
                if val:
                    text_values[entity] = val

        entities: List[Dict[str, Any]] = []

        for entity, value in text_values.items():
            # Support multiple values separated by |
            parts = [p.strip() for p in re.split(r"\s*\|\s*", value) if p.strip()]
            for part in parts:
                char_span = self._find_char_span(original_text, part)
                if not char_span:
                    continue
                start_c, end_c = char_span

                # Map char span to token indices
                start_tok = None
                end_tok = None
                for i, (s, e) in enumerate(token_spans):
                    if start_tok is None and s <= start_c < e:
                        start_tok = i
                    if start_tok is not None and s < end_c <= e:
                        end_tok = i + 1  # exclusive
                        break

                if start_tok is None:
                    # If entity starts between tokens (e.g., punctuation), approximate to nearest token
                    for i, (s, e) in enumerate(token_spans):
                        if s >= start_c:
                            start_tok = i
                            break
                    if start_tok is None:
                        start_tok = 0

                if end_tok is None:
                    for i, (s, e) in enumerate(token_spans[start_tok:], start_tok):
                        if e >= end_c:
                            end_tok = i + 1
                            break
                    if end_tok is None:
                        end_tok = len(tokens)

                if 0 <= start_tok < end_tok <= len(tokens):
                    entities.append({
                        "type": entity,
                        "start": start_tok,
                        "end": end_tok,
                    })

        return entities


def main():
    gen = CSVAnnotationGenerator(config_dir="data/ner/configs")
    out_path = gen.generate_from_csv(
        # csv_path="src/ner/utils/uae_address.csv",
        csv_path="src/ner/utils/validation.csv",
        country_code="uae_xml_roberta_base",
        output_file="data/ner/data/uae_xml_roberta_base/val.jsonl",  # 可省略→默认 data/ner/training_data/uae/generated.json
        # text_column="formatted_address",  # 如不同可自定义
    )
    print(out_path)

if __name__ == "__main__":
    main()