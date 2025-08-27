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

from ner.config.manager import ConfigManager


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
        # Abbreviation/synonym mapping for fuzzy matching on last tokens like Street/St, Road/Rd, etc.
        # Keys should be lowercase canonical forms.
        self._abbrev_synonyms: Dict[str, List[str]] = {
            "street": ["street", "st", "st."],
            "road": ["road", "rd", "rd."],
            "avenue": ["avenue", "ave", "ave."],
            "boulevard": ["boulevard", "blvd", "blvd."],
            "drive": ["drive", "dr", "dr."],
            "lane": ["lane", "ln", "ln."],
            "court": ["court", "ct", "ct."],
            "place": ["place", "pl", "pl."],
            "square": ["square", "sq", "sq."],
            "terrace": ["terrace", "ter", "ter."],
            "highway": ["highway", "hwy", "hwy."],
            "parkway": ["parkway", "pkwy", "pkwy."],
            "building": ["building", "bldg", "bldg."],
            "United Arab Emirates": ["UAE"]
        }

        # Entity priority (higher number = higher priority). Used to resolve overlaps.
        # Tune as needed. Here EMIRATE outranks CITY to avoid city masking emirate in overlaps.
        self._entity_priority: Dict[str, int] = {
            "HOUSE_NUMBER": 100,
            "BUILDING": 90,
            "STREET": 80,
            "SUB_AREA": 70,
            "COMPOUND": 60,
            "EMIRATE": 55,
            "CITY": 50,
            "COUNTRY": 40,
        }

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

        # Create dynamic priority based on CSV column order
        # Higher column index = higher priority (rightmost columns have highest priority)
        dynamic_priority: Dict[str, int] = {}
        for entity, actual_col in entity_to_actual_col.items():
            try:
                col_index = list(df.columns).index(actual_col)
                # Base priority starts from 100, with column index as priority
                # Later columns (higher index) get higher priority
                dynamic_priority[entity.upper()] = 100 + col_index
            except ValueError:
                # Fallback to default priority if column not found
                dynamic_priority[entity.upper()] = self._entity_priority.get(entity.upper(), 0)

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

            labels = self._convert_entities_to_bio_with_priority(tokens, entities, dynamic_priority)
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

    def _find_char_span(self, text: str, value: str) -> Optional[Tuple[int, int]]:
        """Return the first (start,end) char span of value in text using fuzzy matching.

        Strategy:
        - Build a case-insensitive regex from the provided value where known tokens
          (e.g., "Street") expand to alternatives (e.g., "Street|St|St.").
        - Allow flexible separators between tokens (spaces/hyphens).
        - Fallback to exact substring search if regex fails.
        """
        if not isinstance(value, str):
            return None
        s = value.strip()
        if not s:
            return None

        pattern = self._build_value_regex(s)
        m = pattern.search(text)
        if m:
            return m.start(), m.end()

        # Fallback to exact find (case-insensitive)
        idx = text.lower().find(s.lower())
        if idx == -1:
            return None
        return idx, idx + len(s)
    
    def _find_all_char_spans(self, text: str, value: str) -> List[Tuple[int, int]]:
        """Return all (start,end) char spans of value in text using fuzzy matching.

        Strategy:
        - Build a case-insensitive regex from the provided value where known tokens
          (e.g., "Street") expand to alternatives (e.g., "Street|St|St.").
        - Allow flexible separators between tokens (spaces/hyphens).
        - Fallback to exact substring search if regex fails.
        """
        if not isinstance(value, str):
            return []
        s = value.strip()
        if not s:
            return []

        spans = []
        pattern = self._build_value_regex(s)
        
        # Find all regex matches
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end()))
        
        # If no regex matches, fallback to exact substring search
        if not spans:
            text_lower = text.lower()
            s_lower = s.lower()
            start = 0
            while True:
                idx = text_lower.find(s_lower, start)
                if idx == -1:
                    break
                spans.append((idx, idx + len(s)))
                start = idx + 1
        
        return spans

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
                # Find all occurrences of this part in the text
                char_spans = self._find_all_char_spans(original_text, part)
                for start_c, end_c in char_spans:
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

    # --- Helper methods for fuzzy matching and BIO conversion with priority ---

    def _build_value_regex(self, value: str) -> re.Pattern:
        """Build a fuzzy, case-insensitive regex that matches the given value in text.

        - Escapes literal tokens except where a token has known abbreviations/synonyms
          which are expanded into a non-capturing group.
        - Joins tokens with a flexible separator pattern to tolerate spaces/hyphens.
        """
        # Split by whitespace to tokens
        raw_tokens = [t for t in re.split(r"\s+", value.strip()) if t]
        regex_tokens: List[str] = []
        for tok in raw_tokens:
            base = tok.strip().strip(',')
            key = base.lower().rstrip('.')
            if key in self._abbrev_synonyms:
                alts = self._abbrev_synonyms[key]
                # Escape each alt for regex; allow optional trailing dot variants
                escaped_alts = [re.escape(a) for a in alts]
                regex_tokens.append(f"(?:{'|'.join(escaped_alts)})")
            else:
                regex_tokens.append(re.escape(base))

        # Allow spaces and hyphens between tokens
        sep = r"[\s\-]+"
        pattern_str = sep.join(regex_tokens)
        # Use word boundary on ends to avoid partial matches when possible
        pattern_str = rf"\b{pattern_str}\b"
        try:
            return re.compile(pattern_str, flags=re.IGNORECASE)
        except re.error:
            # Extremely defensive: fall back to escaped literal
            return re.compile(re.escape(value), flags=re.IGNORECASE)

    def _convert_entities_to_bio_with_priority(
        self,
        tokens: List[str],
        entities: List[Dict[str, Any]],
        dynamic_priority: Optional[Dict[str, int]] = None,
    ) -> List[str]:
        """Convert entities to BIO labels without overlaps, honoring entity priority.

        - Sort entities by (priority desc, span_length desc) to place higher priority
          and longer entities first.
        - Only place an entity if its entire span is currently unlabeled (all 'O').
        - Uses dynamic_priority if provided, otherwise falls back to self._entity_priority.
        """
        labels: List[str] = ["O"] * len(tokens)

        def priority_for(entity_type: str) -> int:
            if dynamic_priority:
                return dynamic_priority.get(entity_type.upper(), 0)
            return self._entity_priority.get(entity_type.upper(), 0)

        def span_len(ent: Dict[str, Any]) -> int:
            return int(ent.get("end", 0)) - int(ent.get("start", 0))

        sorted_entities = sorted(
            entities,
            key=lambda e: (priority_for(e.get("type", "")), span_len(e)),
            reverse=True,
        )

        for ent in sorted_entities:
            start = int(ent.get("start", 0))
            end = int(ent.get("end", 0))
            etype = str(ent.get("type", "")).upper()
            if not (0 <= start < end <= len(tokens)):
                continue

            # Check if span is free (all 'O')
            if any(label != "O" for label in labels[start:end]):
                continue

            labels[start] = f"B-{etype}"
            for i in range(start + 1, end):
                labels[i] = f"I-{etype}"

        return labels


def main():
    gen = CSVAnnotationGenerator(config_dir="data/ner/configs")
    out_path = gen.generate_from_csv(
        # csv_path="src/ner/utils/uae_address.csv",
        # csv_path="src/ner/utils/train_dataset.csv",
        csv_path="src/ner/utils/evaluate_dataset.csv",
        # csv_path="src/ner/utils/validation.csv",
        # csv_path="src/ner/utils/test_uae_train.csv",
        country_code="uae_xml_roberta_base",
        output_file="data/ner/data/uae_xml_roberta_base/evaluate_50000.jsonl",  # 可省略→默认 data/ner/training_data/uae/generated.json
        # text_column="formatted_address",  # 如不同可自定义
    )
    print(out_path)

if __name__ == "__main__":
    main()