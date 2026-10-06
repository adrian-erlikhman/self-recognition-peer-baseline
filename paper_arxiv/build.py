"""Build the arXiv version: regenerate the arXiv-only numbers and tables,
compile, and check the log.

    python paper_arxiv/build.py            # compile
    python paper_arxiv/build.py --bundle   # also write arxiv_upload.zip

numbers_rev.tex, numbers_study1.tex and the five tables shared with the TACL
submission are frozen copies of the submitted source (Michael's final edits
to three of the tables' captions and widths would be lost by regenerating
them). Everything new for arXiv comes from make_arxiv_numbers.py, which
writes numbers_arxiv.tex and the table_x_*.tex files.

Needs Tectonic on PATH or in $TECTONIC.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# What arXiv gets: the sources, the generated .bbl (arXiv reads it in place of
# running BibTeX, and its name must match main.tex), and the figures.
UPLOAD = ["main.tex", "main.bbl", "acl_natbib.bst",
          "references.bib", "numbers_study1.tex", "numbers_rev.tex",
          "numbers_arxiv.tex"]


def strip_comments(tex: str) -> str:
    """Drop full-line comments and trailing comments (arXiv serves the
    source), keeping escaped percent signs."""
    out = []
    for line in tex.splitlines():
        if line.lstrip().startswith("%"):
            continue
        out.append(re.sub(r"(?<!\\)%.*$", "%" if line.rstrip().endswith("%") else "", line))
    return "\n".join(out) + "\n"


def inputs(tex: Path, seen: set[Path]) -> None:
    """Every .tex file main.tex pulls in, followed recursively."""
    seen.add(tex)
    for name in re.findall(r"\\input\{([^}]+)\}", tex.read_text(encoding="utf-8")):
        f = HERE / (name if name.endswith(".tex") else name + ".tex")
        if f.exists() and f not in seen:
            inputs(f, seen)


def bundle() -> Path:
    used: set[Path] = set()
    inputs(HERE / "main.tex", used)
    files = sorted(set([HERE / f for f in UPLOAD]) | used)
    files += sorted((HERE / "figures").glob("*.pdf"))
    dest = HERE / "arxiv_upload.zip"
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            rel = f.relative_to(HERE).as_posix()
            if f.suffix == ".tex":
                z.writestr(rel, strip_comments(f.read_text(encoding="utf-8")))
            else:
                z.write(f, rel)
    return dest


def plain_abstract() -> str:
    """The abstract with macros expanded and LaTeX removed, for arXiv's
    metadata form (plain text, at most 1,920 characters)."""
    macros: dict[str, str] = {}
    for f in ("numbers_study1.tex", "numbers_rev.tex", "numbers_arxiv.tex"):
        for line in (HERE / f).read_text(encoding="utf-8").splitlines():
            m = re.match(r"\\(?:new|renew|provide)command\{\\([A-Za-z]+)\}\{(.*)\}\s*$", line)
            if m:
                macros[m.group(1)] = m.group(2)
    s = (HERE / "sections" / "abstract.tex").read_text(encoding="utf-8")
    s = re.sub(r"\\([A-Za-z]+)\{\}", lambda m: macros.get(m.group(1), m.group(0)), s)
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = s.replace("\\%", "%").replace("``", '"').replace("''", '"').replace("--", "-")
    s = re.sub(r"\s+", " ", s).strip()
    if "\\" in s:
        raise SystemExit(f"unexpanded LaTeX left in the abstract: {s}")
    return s


def main() -> None:
    gen = HERE / "make_arxiv_numbers.py"
    if gen.exists():
        subprocess.run([sys.executable, str(gen)], check=True)
    tect = os.environ.get("TECTONIC") or shutil.which("tectonic")
    if not tect:
        raise SystemExit("Tectonic not found: set TECTONIC=/path/to/tectonic.exe")
    subprocess.run([tect, "--keep-logs", "--keep-intermediates", "main.tex"], cwd=HERE, check=True)
    log = (HERE / "main.log").read_text(errors="ignore")
    for pat in (r"undefined", r"Overfull \\hbox", r"multiply defined"):
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
    ab = plain_abstract()
    (HERE / "arxiv_abstract.txt").write_text(ab + "\n", encoding="utf-8")
    print(f"abstract: {len(ab)} characters (arXiv limit 1,920), {len(ab.split())} words")
    if "--bundle" in sys.argv:
        print("wrote", bundle())


if __name__ == "__main__":
    main()
