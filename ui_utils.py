from __future__ import annotations

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from schema_utils import module_available



def dataframe_search(df: pd.DataFrame, query: str, columns: list[str] | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if not query:
        return df
    cols = columns or df.columns.tolist()
    mask = pd.Series(False, index=df.index)
    q = query.strip().lower()
    for col in cols:
        if col not in df.columns:
            continue
        mask = mask | df[col].astype(str).str.lower().str.contains(q, na=False)
    return df[mask].copy()



def export_tables_csv_zip(tables: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with ZipFile(output, mode="w", compression=ZIP_DEFLATED) as zf:
        for name, df in tables.items():
            safe_name = str(name).replace("/", "_").replace(" ", "_")
            zf.writestr(f"{safe_name}.csv", df.to_csv(index=False))
    return output.getvalue()



def export_tables_excel_safe(tables: dict[str, pd.DataFrame]) -> tuple[bytes | None, str | None, str | None]:
    engine = None
    if module_available("openpyxl"):
        engine = "openpyxl"
    elif module_available("xlsxwriter"):
        engine = "xlsxwriter"
    if engine is None:
        return None, None, "Excel export is unavailable because neither openpyxl nor xlsxwriter is installed. CSV ZIP export is available instead."

    output = BytesIO()
    with pd.ExcelWriter(output, engine=engine) as writer:
        for sheet_name, df in tables.items():
            df.to_excel(writer, sheet_name=str(sheet_name)[:31], index=False)
    return output.getvalue(), engine, None
