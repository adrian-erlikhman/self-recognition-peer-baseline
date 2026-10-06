"""Build the TACL draft: regenerate numbers and figures, then compile.

    python paper_tacl/build.py

Needs Tectonic on PATH or in $TECTONIC. Reports the page on which the
references begin, since TACL allows 10 content pages before them.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    subprocess.run([sys.executable, str(HERE / "make_numbers.py")], check=True)
    fig = HERE / "make_figures.py"
    if fig.exists():
        subprocess.run([sys.executable, str(fig)], check=True)
    tect = os.environ.get("TECTONIC") or shutil.which("tectonic")
    if not tect:
        raise SystemExit("Tectonic not found: set TECTONIC=/path/to/tectonic.exe")
    subprocess.run([tect, "--keep-logs", "main.tex"], cwd=HERE, check=True)
    log = (HERE / "main.log").read_text(errors="ignore")
    for pat in (r"undefined", r"Overfull \\hbox"):
        hits = sorted(set(re.findall(rf".*{pat}.*", log)))
        if hits:
            print(f"{len(hits)} log lines matching {pat!r}:")
            for h in hits[:12]:
                print("   ", h.strip()[:140])
    try:
        from pypdf import PdfReader
        r = PdfReader(str(HERE / "main.pdf"))
        ref_page = next((i + 1 for i, p in enumerate(r.pages)
                         if re.search(r"^\s*References\s*$", p.extract_text() or "", re.M)), None)
        print(f"{len(r.pages)} pages; references start on page {ref_page}")
    except Exception as exc:  # noqa: BLE001
        print(f"page check skipped: {exc}")


if __name__ == "__main__":
    main()
