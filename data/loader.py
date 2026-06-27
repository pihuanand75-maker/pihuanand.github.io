"""
Multi-format data file loader.
Supports: CSV, TSV, Excel (.xlsx/.xls), JSON, Parquet, JSON Lines.
Returns: dict mapping table_name → pd.DataFrame
All errors raise DataLoadError with user-friendly messages.
"""
import io
import pandas as pd
from pathlib import Path
from typing import Union


class DataLoadError(Exception):
    pass


class DataLoader:
    SUPPORTED_EXTENSIONS = {'.csv', '.tsv', '.xlsx', '.xls', '.json', '.parquet', '.jsonl'}

    @classmethod
    def load_file(cls, file_obj, filename: str) -> dict[str, pd.DataFrame]:
        """
        Load file object into a dict of DataFrames.
        file_obj: Streamlit UploadedFile or file-like object with .read()
        filename: original filename (used to detect format)
        Returns: {"table_name": pd.DataFrame}
        """
        ext = Path(filename).suffix.lower()
        stem = Path(filename).stem.lower()
        # Sanitize table name: remove non-alphanumeric except underscore
        table_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in stem)
        if table_name and table_name[0].isdigit():
            table_name = 't_' + table_name
        if not table_name:
            table_name = 'data_table'

        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise DataLoadError(
                f"Unsupported file type: '{ext}'. "
                f"Supported: {', '.join(sorted(cls.SUPPORTED_EXTENSIONS))}"
            )

        try:
            raw_bytes = file_obj.read()
            file_like = io.BytesIO(raw_bytes)

            if ext == '.csv':
                df = cls._load_csv(file_like, raw_bytes)
            elif ext == '.tsv':
                df = pd.read_csv(file_like, sep='\t', encoding='utf-8-sig')
            elif ext in ('.xlsx', '.xls'):
                return cls._load_excel(file_like, ext, stem)
            elif ext == '.json':
                df = cls._load_json(file_like)
            elif ext == '.jsonl':
                df = pd.read_json(file_like, lines=True)
            elif ext == '.parquet':
                df = pd.read_parquet(file_like)
            else:
                raise DataLoadError(f"Loader not implemented for: {ext}")

        except DataLoadError:
            raise
        except Exception as e:
            raise DataLoadError(f"Failed to parse '{filename}': {str(e)}")

        df = cls._clean_dataframe(df)
        return {table_name: df}

    @classmethod
    def _load_csv(cls, file_like: io.BytesIO, raw_bytes: bytes) -> pd.DataFrame:
        """Auto-detect CSV encoding and separator."""
        for encoding in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
            try:
                file_like.seek(0)
                sample = file_like.read(4096).decode(encoding)
                first_line = sample.split('\n')[0]
                sep = ','
                if first_line.count(';') > first_line.count(','):
                    sep = ';'
                elif first_line.count('\t') > first_line.count(','):
                    sep = '\t'
                file_like.seek(0)
                return pd.read_csv(io.BytesIO(raw_bytes), sep=sep, encoding=encoding)
            except Exception:
                continue
        raise DataLoadError("Could not decode CSV file. Try saving as UTF-8.")

    @classmethod
    def _load_excel(cls, file_like: io.BytesIO, ext: str, stem: str) -> dict[str, pd.DataFrame]:
        """Load all sheets from an Excel file. Returns multi-table dict."""
        engine = 'openpyxl' if ext == '.xlsx' else 'xlrd'
        xls = pd.ExcelFile(file_like, engine=engine)
        tables = {}
        for sheet_name in xls.sheet_names:
            file_like.seek(0)
            df = pd.read_excel(file_like, sheet_name=sheet_name, engine=engine)
            df = cls._clean_dataframe(df)
            if len(df) > 0:
                safe_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in sheet_name.lower())
                if safe_name and safe_name[0].isdigit():
                    safe_name = 't_' + safe_name
                if not safe_name:
                    safe_name = f'sheet_{len(tables)}'
                tables[safe_name] = df
        return tables

    @classmethod
    def _load_json(cls, file_like: io.BytesIO) -> pd.DataFrame:
        """Handle both JSON array and nested JSON object formats."""
        import json
        data = json.load(file_like)
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, list) and len(val) > 0:
                    return pd.DataFrame(val)
            return pd.json_normalize(data)
        raise DataLoadError("JSON format not recognized. Expected array or object with data array.")

    @classmethod
    def _clean_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame: strip column names, drop fully empty rows/cols."""
        df.columns = [
            ''.join(c if c.isalnum() or c == '_' else '_' for c in str(col).strip().lower())
            for col in df.columns
        ]
        df.columns = [f"col_{c}" if c and c[0].isdigit() else (c if c else f"col_{i}") for i, c in enumerate(df.columns)]
        # Handle duplicate column names
        seen = {}
        new_cols = []
        for col in df.columns:
            if col in seen:
                seen[col] += 1
                new_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_cols.append(col)
        df.columns = new_cols
        df = df.dropna(how='all').dropna(axis=1, how='all')
        df = df.reset_index(drop=True)
        return df
