"""Assemble, check and compile the IEEE-style report.

* orders \\bibitem entries by first \\cite appearance (IEEE numbering);
* fails if any cited key lacks a reference or any reference is uncited;
* counts body words (Introduction..Conclusion, excluding captions, tables,
  references and appendix) and fails outside 3,500-5,000;
* runs pdflatex twice and copies the PDF to reports/report.pdf.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SECTIONS = ["01_introduction", "02_literature", "03_methodology", "04_results", "05_discussion", "06_conclusion"]


def body_text(tex: str) -> str:
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    tex = re.sub(r"\\begin\{(table\*?|figure\*?|tabular|equation)\}.*?\\end\{\1\}", " ", tex, flags=re.S)
    tex = re.sub(r"\\input\{[^}]*\}", " ", tex)
    tex = re.sub(r"\\(cite|label|ref|includegraphics)\{[^}]*\}", " ", tex)
    tex = re.sub(r"\\(section|subsection)\*?\{([^}]*)\}", r" \2 ", tex)
    tex = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", tex)
    tex = re.sub(r"[{}$~\\]", " ", tex)
    return tex


def main() -> int:
    texts = {s: (HERE / "sections" / f"{s}.tex").read_text() for s in SECTIONS}
    appendix = (HERE / "sections" / "07_appendix.tex").read_text()
    cites = []
    for s in SECTIONS:
        for group in re.findall(r"\\cite\{([^}]*)\}", texts[s]):
            for key in group.split(","):
                key = key.strip()
                if key not in cites:
                    cites.append(key)
    for group in re.findall(r"\\cite\{([^}]*)\}", appendix):
        for key in group.split(","):
            if key.strip() not in cites:
                cites.append(key.strip())
    refs = {}
    for block in re.split(r"(?=\\bibitem\{)", (HERE / "references.tex").read_text()):
        m = re.match(r"\\bibitem\{([^}]*)\}", block)
        if m:
            refs[m.group(1)] = block.strip()
    missing = [k for k in cites if k not in refs]
    uncited = [k for k in refs if k not in cites]
    if missing or uncited:
        print("Missing references:", missing, "| Uncited references:", uncited)
        return 1
    (HERE / "references_ordered.tex").write_text("\n\n".join(refs[k] for k in cites) + "\n")

    words = {s: len(body_text(texts[s]).split()) for s in SECTIONS}
    total = sum(words.values())
    print("Word counts:", words, "TOTAL", total, "| references:", len(cites))
    if not 3500 <= total <= 5000:
        print("Body word count outside 3,500-5,000")
        return 1
    (HERE / "word_count.txt").write_text(f"{total}\n" + "\n".join(f"{k}: {v}" for k, v in words.items()) + "\n")

    for _ in range(2):
        res = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
                             cwd=HERE, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout[-3000:])
            return 1
    warnings = [l for l in res.stdout.splitlines() if "undefined" in l.lower() or "Overfull \\hbox" in l]
    print("LaTeX warnings:", warnings[:10])
    shutil.copy(HERE / "report.pdf", HERE.parent / "reports" / "report.pdf")
    print("PDF written to reports/report.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
