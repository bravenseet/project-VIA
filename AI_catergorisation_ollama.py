import pandas as pd
import ollama
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

 
# ── Config ────────────────────────────────────────────────────────────────────
RAW_FILE    = "raw_file.xlsx"
OUTPUT_FILE = "output.xlsx"
AI_MODEL    = "gemma2:2b"   # small & fast; alternatives: "llama3.2:1b", "phi3:mini"

MAX_WORKERS = 6
# Only rows with these statuses will be included in the output
ALLOWED_STATUSES = ["Approved / Started", "Verified", "Completed"]
 
# ── Prompt Template ───────────────────────────────────────────────────────────
CATEGORY_SYSTEM_PROMPT = """
You are a classifier for student VIA (Values In Action) activities.

Your task is to read an activity description and classify it into ONE (and only one) of the categories listed below.

Return ONLY the category name exactly as written.
Do NOT include explanations, punctuation, or extra words.
Do NOT rephrase or combine categories.

Choose the category that BEST matches the MAIN role or task performed by the student (not the organisation’s mission).

If multiple activities are mentioned, prioritise the student’s primary responsibility.

Categories:
- Administrative and Logistics
- Befriending
- Organising and Facilitating
- Mentoring/Tutoring
- Fundraising
- Charity Event

Classification guidelines:
- Administrative and Logistics: Packing, sorting, distributing items, door-knocking, data collection, coordination support.
- Befriending: One-on-one care, companionship, supervising or engaging beneficiaries (e.g. elderly, children, patients).
- Organising and Facilitating: Leading, managing events, directing crowds, coordinating volunteers, handling operations.
- Mentoring/Tutoring: Teaching, coaching, guiding academic work or structured learning activities.
- Fundraising: Raising money through campaigns, challenges, or donation drives.
- Charity Event: Helping out at public events or booths (e.g. selling items, welcoming visitors, event support).

Examples:
(These are for understanding only — DO NOT output them.)

Input: "I helped primary school students with their homework and explained math concepts."
Output: Mentoring/Tutoring

Input: "I packed and sorted donated clothes before distributing them."
Output: Administrative and Logistics

Input: "I was paired with a child at a camp and took care of her daily needs."
Output: Befriending

Input: "I managed volunteers and directed crowd flow during an event."
Output: Organising and Facilitating

Input: "I participated in a running challenge to raise money for charity."
Output: Fundraising

Input: "I helped run a drinks stall and welcomed visitors at a charity event."
Output: Charity Event

If you want, I can make this even stricter (for better LLM accuracy) or adapt it for batch classification.
"""
 
def categorize_one(index: int, description: str) -> tuple:
    """Classify a single description. Returns (index, category) to preserve order."""
    if not isinstance(description, str) or description.strip() == "":
        return (index, "")
 
    response = ollama.chat(
        model=AI_MODEL,
        messages=[
            {"role": "system", "content": CATEGORY_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Activity description:\n{description.strip()}"}
        ]
    )
    return (index, response["message"]["content"].strip())
 
 
# ── Read RAW file ─────────────────────────────────────────────────────────────
def read_raw(filepath: str) -> pd.DataFrame:
    """
    RAW portal export has two header rows:
      row 0 -> metadata banner (skipped)
      row 1 -> actual column names
    """
    df = pd.read_excel(filepath, header=1, skiprows=[0])
    df.columns = [c.replace("\xa0", " ").strip() for c in df.columns]
    return df
 
 
# ── Filter rows ───────────────────────────────────────────────────────────────
def filter_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with an allowed status and a non-blank description."""
    total = len(raw)
    raw = raw[raw["Status"].isin(ALLOWED_STATUSES)]
    raw = raw[raw["Description"].notna() & (raw["Description"].str.strip() != "")]
    print(f"  {total} total records -> {len(raw)} kept "
          f"(status: {', '.join(ALLOWED_STATUSES)} + non-blank description)")
    return raw.copy()
 
 
# ── Column mapping ────────────────────────────────────────────────────────────
def map_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Map RAW portal columns -> 9 target columns:
        Activity Year | Class | Name | VIA Activity | Activity Description |
        Category | Activity Type | Role | Hours
    """
    out = pd.DataFrame()
 
    out["Activity Year"]        = pd.to_datetime(raw["Initiated At"], errors="coerce").dt.year.astype("Int64")
    out["Class"]                = ""
    out["Name"]                 = raw["Initiator"]
    out["VIA Activity"]         = raw["Name of Activity"]
    out["Activity Description"] = raw["Description"]
    out["Category"]             = ""                # populated by AI step below
    out["Activity Type"]        = raw["Type"]
    out["Role"]                 = ""
    out["Hours"]                = pd.to_numeric(raw["Duration (hrs)"], errors="coerce").astype("Int64")
 
    return out
 
 
# ── AI categorisation step (threaded) ────────────────────────────────────────
def add_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Classify all descriptions in parallel using threads."""
    descriptions = df["Activity Description"].tolist()
    total        = len(descriptions)
    results      = {}
    completed    = 0
 
    print(f"Categorising {total} rows using {MAX_WORKERS} parallel threads (Ollama/{AI_MODEL})...")
 
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(categorize_one, i, desc): i
            for i, desc in enumerate(descriptions)
        }
        for future in as_completed(futures):
            index, category = future.result()
            results[index] = category
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"  {completed}/{total} done")
 
    # Reassemble in original order
    df["Category"] = [results[i] for i in range(total)]
    return df
 
 
# ── Write output ──────────────────────────────────────────────────────────────
def write_output(df: pd.DataFrame, filepath: str) -> None:
    # Convert pandas Int64 (nullable) to plain int/None so openpyxl doesn't
    # emit "data was lost" warnings for rows where the value is pd.NA
    df = df.copy()
    for col in df.select_dtypes("Int64").columns:
        df[col] = df[col].astype(object).where(df[col].notna(), other=None)

    # Prepend a space to strings starting with formula characters so openpyxl
    # writes them as plain strings, not formula nodes, avoiding Excel repair errors
    def _esc(val):
        if isinstance(val, str) and val.startswith(("=", "+", "-", "@")):
            return " " + val
        return val
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].apply(_esc)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="via_all_2025", index=False)

        ws = writer.sheets["via_all_2025"]
        col_widths = {
            "A": 12,  # Activity Year
            "B": 10,  # Class
            "C": 25,  # Name
            "D": 35,  # VIA Activity
            "E": 60,  # Activity Description
            "F": 30,  # Category
            "G": 15,  # Activity Type
            "H": 20,  # Role
            "I": 8,   # Hours
        }
        for col_letter, width in col_widths.items():
            ws.column_dimensions[col_letter].width = width

    print(f"Output saved to {filepath}")
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    start_time = time.perf_counter()

    print("Reading RAW portal file...")
    raw = read_raw(RAW_FILE)
    print(f"  {len(raw)} records loaded.")
 
    print("Filtering by status and description...")
    raw = filter_rows(raw)
 
    print("Mapping columns...")
    df = map_columns(raw)
 
    # Comment out the next line if you want to skip AI and fill Category manually
    df = add_categories(df)
 
    print("Sorting by name...")
    df = df.sort_values("Name", ascending=True, ignore_index=True)
 
    print("Writing output...")
    write_output(df, OUTPUT_FILE)
    end_time = time.perf_counter()
    print(f"Elapsed time: {end_time - start_time:.4f} seconds")
 
 
if __name__ == "__main__":
    main()