"""
VIA Activity Categoriser — GUI Launcher
Run with: pythonw launcher.pyw  (no console window)
       or: python launcher.pyw
Create a Windows shortcut pointing to: pythonw "<full path>\launcher.pyw"
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys

# ── Shared processing imports ─────────────────────────────────────────────────
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Constants ─────────────────────────────────────────────────────────────────
ALLOWED_STATUSES = ["Approved / Started", "Verified", "Completed"]
MAX_WORKERS = 6
OLLAMA_MODEL_DEFAULT = "gemma2:2b"
GEMINI_MODEL_DEFAULT = "gemini-2.0-flash"

CATEGORY_SYSTEM_PROMPT = """
You are a classifier for student VIA (Values In Action) activities.

Your task is to read an activity description and classify it into ONE (and only one) of the categories listed below.

Return ONLY the category name exactly as written.
Do NOT include explanations, punctuation, or extra words.
Do NOT rephrase or combine categories.

Choose the category that BEST matches the MAIN role or task performed by the student (not the organisation's mission).

If multiple activities are mentioned, prioritise the student's primary responsibility.

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
"""


# ── Data pipeline ─────────────────────────────────────────────────────────────

def read_raw(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath, header=1, skiprows=[0])
    df.columns = [c.replace("\xa0", " ").strip() for c in df.columns]
    return df


def filter_rows(raw: pd.DataFrame) -> pd.DataFrame:
    total = len(raw)
    raw = raw[raw["Status"].isin(ALLOWED_STATUSES)]
    raw = raw[raw["Description"].notna() & (raw["Description"].str.strip() != "")]
    return raw.copy(), total, len(raw)


def map_columns(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Activity Year"]        = pd.to_datetime(raw["Initiated At"], errors="coerce").dt.year.astype("Int64")
    out["Class"]                = ""
    out["Name"]                 = raw["Initiator"]
    out["VIA Activity"]         = raw["Name of Activity"]
    out["Activity Description"] = raw["Description"]
    out["Category"]             = ""
    out["Activity Type"]        = raw["Type"]
    out["Role"]                 = ""
    out["Hours"]                = pd.to_numeric(raw["Duration (hrs)"], errors="coerce").astype("Int64")
    return out


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
        col_widths = {"A": 12, "B": 10, "C": 25, "D": 35,
                      "E": 60, "F": 30, "G": 15, "H": 20, "I": 8}
        for col, width in col_widths.items():
            ws.column_dimensions[col].width = width


# ── AI backends ───────────────────────────────────────────────────────────────

def make_ollama_categorizer(model: str):
    import ollama
    def categorize_one(index: int, description: str) -> tuple:
        if not isinstance(description, str) or description.strip() == "":
            return (index, "")
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": CATEGORY_SYSTEM_PROMPT},
                {"role": "user",   "content": f"Activity description:\n{description.strip()}"}
            ]
        )
        return (index, response["message"]["content"].strip())
    return categorize_one


def make_gemini_categorizer(model: str, api_key: str):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    _model = genai.GenerativeModel(
        model_name=model,
        system_instruction=CATEGORY_SYSTEM_PROMPT,
    )
    def categorize_one(index: int, description: str) -> tuple:
        if not isinstance(description, str) or description.strip() == "":
            return (index, "")
        response = _model.generate_content(f"Activity description:\n{description.strip()}")
        return (index, response.text.strip())
    return categorize_one


def add_categories(df: pd.DataFrame, categorize_one, log_fn) -> pd.DataFrame:
    descriptions = df["Activity Description"].tolist()
    total = len(descriptions)
    results = {}
    completed = 0

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
                log_fn(f"  {completed}/{total} categorised")

    df["Category"] = [results[i] for i in range(total)]
    return df


# ── GUI ───────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VIA Activity Categoriser")
        self.resizable(False, False)
        self._build_ui()
        self._on_backend_change()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        # ── File paths ──────────────────────────────────────────────────────
        files_frame = ttk.LabelFrame(self, text="Files", padding=10)
        files_frame.grid(row=0, column=0, sticky="ew", **pad)

        ttk.Label(files_frame, text="Input file (.xlsx):").grid(row=0, column=0, sticky="w")
        self.input_var = tk.StringVar()
        ttk.Entry(files_frame, textvariable=self.input_var, width=48).grid(row=0, column=1, padx=(6, 4))
        ttk.Button(files_frame, text="Browse…", command=self._browse_input).grid(row=0, column=2)

        ttk.Label(files_frame, text="Output file (.xlsx):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.output_var = tk.StringVar()
        ttk.Entry(files_frame, textvariable=self.output_var, width=48).grid(row=1, column=1, padx=(6, 4), pady=(6, 0))
        ttk.Button(files_frame, text="Browse…", command=self._browse_output).grid(row=1, column=2, pady=(6, 0))

        # ── Backend ─────────────────────────────────────────────────────────
        backend_frame = ttk.LabelFrame(self, text="AI Backend", padding=10)
        backend_frame.grid(row=1, column=0, sticky="ew", **pad)

        self.backend_var = tk.StringVar(value="ollama")
        ttk.Radiobutton(backend_frame, text="Ollama (local, no API key needed)",
                        variable=self.backend_var, value="ollama",
                        command=self._on_backend_change).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(backend_frame, text="Gemini (Google API)",
                        variable=self.backend_var, value="gemini",
                        command=self._on_backend_change).grid(row=1, column=0, sticky="w", pady=(4, 0))

        # Ollama model
        self.ollama_frame = ttk.Frame(backend_frame)
        self.ollama_frame.grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Label(self.ollama_frame, text="  Model:").pack(side="left")
        self.ollama_model_var = tk.StringVar(value=OLLAMA_MODEL_DEFAULT)
        ttk.Entry(self.ollama_frame, textvariable=self.ollama_model_var, width=20).pack(side="left", padx=(4, 0))

        # Gemini model + key
        self.gemini_frame = ttk.Frame(backend_frame)
        self.gemini_frame.grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Label(self.gemini_frame, text="  Model:").grid(row=0, column=0, sticky="w")
        self.gemini_model_var = tk.StringVar(value=GEMINI_MODEL_DEFAULT)
        ttk.Entry(self.gemini_frame, textvariable=self.gemini_model_var, width=22).grid(row=0, column=1, padx=(4, 0))
        ttk.Label(self.gemini_frame, text="  API Key:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.api_key_var = tk.StringVar(value=os.environ.get("GEMINI_API_KEY", ""))
        ttk.Entry(self.gemini_frame, textvariable=self.api_key_var, width=44, show="*").grid(
            row=1, column=1, padx=(4, 0), pady=(4, 0))

        # ── Run button ──────────────────────────────────────────────────────
        self.run_btn = ttk.Button(self, text="▶  Run Categorisation", command=self._run)
        self.run_btn.grid(row=2, column=0, pady=(0, 4))

        # ── Log ─────────────────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(self, text="Log", padding=6)
        log_frame.grid(row=3, column=0, sticky="nsew", **pad)
        self.log_box = scrolledtext.ScrolledText(log_frame, width=72, height=14,
                                                  state="disabled", font=("Consolas", 9))
        self.log_box.pack(fill="both", expand=True)

    def _on_backend_change(self):
        if self.backend_var.get() == "ollama":
            self.ollama_frame.grid()
            self.gemini_frame.grid_remove()
        else:
            self.ollama_frame.grid_remove()
            self.gemini_frame.grid()

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="Select input file",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if path:
            self.input_var.set(path)
            if not self.output_var.get():
                default_out = os.path.join(os.path.dirname(path), "output.xlsx")
                self.output_var.set(default_out)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Save output file as",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if path:
            self.output_var.set(path)

    def _log(self, msg: str):
        self.log_box.config(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")
        self.update_idletasks()

    def _run(self):
        input_path  = self.input_var.get().strip()
        output_path = self.output_var.get().strip()
        backend     = self.backend_var.get()

        if not input_path:
            messagebox.showerror("Missing input", "Please select an input file.")
            return
        if not output_path:
            messagebox.showerror("Missing output", "Please specify an output file path.")
            return
        if backend == "gemini" and not self.api_key_var.get().strip():
            messagebox.showerror("Missing API key", "Please enter your Gemini API key.")
            return

        self.run_btn.config(state="disabled")
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.config(state="disabled")

        thread = threading.Thread(target=self._process,
                                  args=(input_path, output_path, backend),
                                  daemon=True)
        thread.start()

    def _process(self, input_path, output_path, backend):
        try:
            self._log("Reading input file...")
            raw = read_raw(input_path)
            self._log(f"  {len(raw)} records loaded.")

            self._log("Filtering by status and description...")
            raw, total_before, total_after = filter_rows(raw)
            self._log(f"  {total_before} total -> {total_after} kept.")

            self._log("Mapping columns...")
            df = map_columns(raw)

            if backend == "ollama":
                model = self.ollama_model_var.get().strip()
                self._log(f"Categorising with Ollama / {model}...")
                try:
                    categorize_one = make_ollama_categorizer(model)
                except ImportError:
                    self._log("ERROR: ollama package not installed. Run: pip install ollama")
                    self.run_btn.config(state="normal")
                    return
            else:
                model   = self.gemini_model_var.get().strip()
                api_key = self.api_key_var.get().strip()
                self._log(f"Categorising with Gemini / {model}...")
                try:
                    categorize_one = make_gemini_categorizer(model, api_key)
                except ImportError:
                    self._log("ERROR: google-generativeai not installed. Run: pip install google-generativeai")
                    self.run_btn.config(state="normal")
                    return

            df = add_categories(df, categorize_one, self._log)

            self._log("Sorting by name...")
            df = df.sort_values("Name", ascending=True, ignore_index=True)

            self._log(f"Writing output to {output_path}...")
            write_output(df, output_path)

            self._log("\nDone!")
            self.after(0, lambda: messagebox.showinfo(
                "Complete",
                f"Categorisation complete!\n\n{total_after} records processed.\nOutput saved to:\n{output_path}"
            ))

        except Exception as e:
            self._log(f"\nERROR: {e}")
            self.after(0, lambda: messagebox.showerror("Error", str(e)))

        finally:
            self.after(0, lambda: self.run_btn.config(state="normal"))


if __name__ == "__main__":
    app = App()
    app.mainloop()
