#!/usr/bin/env python3
"""Build the FedMPS capstone thesis from docs/thesis/content.

Uses docs/CAPSTONE_THESIS_Format.docx as the template, so the styles, fonts,
page size and page-number footer are the format's own. The body of the
template is removed and rebuilt from:

    content/meta.json        title page and certificate details
    content/00_front.md      acknowledgements and abstract
    content/NN_*.md          one file per chapter, in file-name order
    content/references.md    "@key: reference text" lines; the References
                             chapter is generated from the keys cited

Markdown subset understood here:

    # CHAPTER 1 | INTRODUCTION         chapter (starts a new page)
    ## 1.1 Objectives                  section
    ### 1.1.1 Scope                    sub-section
    #### Plain bold heading            unnumbered minor heading
    paragraph text                     blank line between paragraphs
    - item / 1. item                   bulleted / numbered list
    ![Caption](path){#fig:id w=6}      figure, numbered through the thesis
    Table: Caption {#tab:id}           table caption, followed by a | table |
    ```                                code block
    !code[path/to/file.py](Name)       code block with the real source of a
                                       function, class or Class.method
    $$ text $$ {#eq:id}                numbered equation, "(2.1)" at the right margin
    {fig:id} {tab:id} {eq:id}          cross-reference -> "Figure 3" / "Table 2.1" /
                                       "Equation 2.1"
    [@key] or [@a, @b]                 citation -> [n]; numbered by first use, and
                                       consecutive numbers joined: [1-3],[5]
    **bold**  *italic*  `code`         inline formatting

Page numbers in the table of contents and the lists of tables and figures
are filled in by rendering the document to PDF with Word, reading where
each heading and caption landed, and building again.

Usage:
    python docs/thesis/build_thesis.py            # full build: .docx + .pdf
    python docs/thesis/build_thesis.py --no-pdf   # .docx only, no page numbers
"""
import sys as _sys

for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import copy
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import docx
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_TAB_ALIGNMENT
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEMPLATE = ROOT / "docs" / "CAPSTONE_THESIS_Format.docx"
CONTENT = HERE / "content"
OUT_DOCX = ROOT / "docs" / "FedMPS_Capstone_Thesis.docx"
OUT_PDF = OUT_DOCX.with_suffix(".pdf")
LOGO = HERE / "assets" / "vitap_logo.jpeg"
# Per-document style, from "style" in the content folder's meta.json:
#   equations      "chapter" -> "(2.1)";  "section" -> "Eqn 2.2.1" (chapter.section.n)
#   table_align    "auto" -> numbers centred;  "left" -> every body cell left-aligned
#   references     "chapter" -> listed as the last chapter;  "unnumbered" -> no number
#   abbreviations  file of "ABBR: Full Form" lines; the first use in running text
#                  is written out as "Full Form (ABBR)", every later use stays short
#   lint           true -> warn about the writing rules of the v2 document
STYLE: Dict[str, object] = {}

BODY_FONT = "Times New Roman"
CODE_FONT = "Consolas"
TEXT_WIDTH_IN = 6.3          # usable width with the margins set below
LANDSCAPE_WIDTH_IN = 8.9     # usable width of a landscape page, same margins
PLACEHOLDER = "00"
# Cells that hold a number (or a dash) are centred in tables; text is left-aligned.
NUMERIC = re.compile(r"^([+\-−]?[\d.,]+\s*(%|×|x|s|min|MB|GB|KB)?(\s*±\s*[\d.]+)?|—|-|n/a)$")
_UNITS = re.compile(r"\b(min|s|h|ms|MB|GB|KB|bytes?|px|fps|of|vs|M|k|x|bits?|rounds?|"
                    r"frames?|people|images|identities|faces|per|million|"
                    r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b|US\$")


def is_numeric(cell: str) -> bool:
    """A number, possibly with units, ranges, ± or a ratio such as '1,988 of 2,088'."""
    c = cell.strip("* ").strip()
    if NUMERIC.match(c) or c in ("∞", "—", "-", "n/a", ""):
        return bool(c)
    if not re.search(r"\d", c):
        return False
    rest = _UNITS.sub("", c)
    return not re.sub(r"[\d\s.,+\-−–±%×/:≈~<>=$₹⁰¹²³⁴⁵⁶⁷⁸⁹⁻·@]", "", rest)


def capitalise(cell: str) -> str:
    """Start a text cell with a capital letter (code, symbols and roman page numbers
    are left alone)."""
    if re.fullmatch(r"[ivxlc]+(<br>[ivxlc0-9]+)*", cell):
        return cell
    m = re.match(r"^(\*{0,2})([a-z])", cell)
    return cell if not m else m.group(1) + m.group(2).upper() + cell[m.end():]


def log(msg: str) -> None:
    print(f"[thesis] {msg}", flush=True)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

@dataclass
class Block:
    kind: str                 # chapter, section, subsection, minor, para,
                              # bullets, numbers, figure, table, code, display, pagebreak
    text: str = ""
    items: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    path: str = ""
    width: float = 6.0
    landscape: bool = False   # figure on its own landscape page
    label: str = ""
    number: str = ""          # filled in by numbering


@dataclass
class Chapter:
    number: int
    title: str
    blocks: List[Block]


CODE_RE = re.compile(r"^!code\[(?P<path>[^\]]+)\]\((?P<obj>[\w.]+)\)\s*$")


def source_of(path: str, obj: str) -> str:
    """Exact source of a top-level function or class, or Class.method."""
    import ast
    import textwrap
    src = (ROOT / path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    parts = obj.split(".")
    nodes = tree.body
    found = None
    for name in parts:
        found = next((n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                      and n.name == name), None)
        if found is None:
            raise SystemExit(f"!code: {obj} not found in {path}")
        nodes = found.body
    lines = src.splitlines()
    start = (found.decorator_list[0].lineno if found.decorator_list else found.lineno) - 1
    return textwrap.dedent("\n".join(lines[start:found.end_lineno]))


FIG_RE = re.compile(r"^!\[(?P<cap>.*?)\]\((?P<path>[^)]+)\)(?:\{(?P<opts>[^}]*)\})?\s*$")
TAB_RE = re.compile(r"^Table:\s*(?P<cap>.*?)\s*(?:\{#(?P<label>[\w:.-]+)\})?\s*$")


def parse_opts(opts: Optional[str]) -> Tuple[str, float, bool]:
    label, width, landscape = "", 6.0, False
    for tok in (opts or "").split():
        if tok.startswith("#"):
            label = tok[1:]
        elif tok.startswith("w="):
            width = float(tok[2:])
        elif tok == "landscape":
            landscape = True
    return label, width, landscape


def parse_markdown(text: str) -> List[Block]:
    lines = text.splitlines()
    blocks: List[Block] = []
    para: List[str] = []
    i = 0

    def flush_para():
        if para:
            blocks.append(Block("para", " ".join(s.strip() for s in para)))
            para.clear()

    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.startswith("<!--"):
            while "-->" not in lines[i]:
                i += 1
            i += 1
            continue
        if not s:
            flush_para()
            i += 1
            continue
        if s.startswith("```"):
            flush_para()
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i].rstrip("\n"))
                i += 1
            blocks.append(Block("code", "\n".join(code)))
            i += 1
            continue
        if s == "\\pagebreak":
            flush_para()
            blocks.append(Block("pagebreak"))
            i += 1
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            flush_para()
            level, body = len(m.group(1)), m.group(2).strip()
            kind = {1: "chapter", 2: "section", 3: "subsection", 4: "minor"}[level]
            blocks.append(Block(kind, body))
            i += 1
            continue
        m = CODE_RE.match(s)
        if m:
            flush_para()
            blocks.append(Block("code", source_of(m.group("path"), m.group("obj"))))
            i += 1
            continue
        m = FIG_RE.match(s)
        if m:
            flush_para()
            label, width, landscape = parse_opts(m.group("opts"))
            blocks.append(Block("figure", m.group("cap"), path=m.group("path"),
                                width=width, label=label, landscape=landscape))
            i += 1
            continue
        m = TAB_RE.match(s)
        if m:
            flush_para()
            cap, label = m.group("cap"), m.group("label") or ""
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            blocks.append(Block("table", cap, rows=rows, label=label))
            continue
        m = re.match(r"^\$\$(.+?)\$\$\s*(?:\{#(eq:[\w.-]+)\})?\s*$", s)
        if m:
            flush_para()
            blocks.append(Block("display", m.group(1).strip(), label=m.group(2) or ""))
            i += 1
            continue
        if re.match(r"^[-*]\s+", s) or re.match(r"^\d+\.\s+", s):
            flush_para()
            numbered = bool(re.match(r"^\d+\.\s+", s))
            items = []
            while i < len(lines):
                t = lines[i].strip()
                if numbered and re.match(r"^\d+\.\s+", t):
                    items.append(re.sub(r"^\d+\.\s+", "", t))
                elif not numbered and re.match(r"^[-*]\s+", t):
                    items.append(re.sub(r"^[-*]\s+", "", t))
                elif t and lines[i].startswith(("  ", "\t")) and items:
                    items[-1] += " " + t          # continuation line
                else:
                    break
                i += 1
            blocks.append(Block("numbers" if numbered else "bullets", items=items))
            continue
        para.append(line)
        i += 1
    flush_para()
    return blocks


def load_chapters() -> List[Chapter]:
    chapters: List[Chapter] = []
    for path in sorted(CONTENT.glob("[0-9][0-9]*_*.md")):
        if path.name.startswith("00_"):
            continue
        blocks = parse_markdown(path.read_text(encoding="utf-8"))
        for b in blocks:
            if b.kind == "chapter":
                m = re.match(r"CHAPTER\s+(\d+)\s*\|\s*(.+)", b.text, re.I)
                if not m:
                    raise SystemExit(f"{path.name}: chapter heading must be "
                                     f"'# CHAPTER N | TITLE', got {b.text!r}")
                chapters.append(Chapter(int(m.group(1)), m.group(2).strip(), []))
            elif chapters:
                chapters[-1].blocks.append(b)
            else:
                raise SystemExit(f"{path.name}: text before the first chapter heading")
    return chapters


def number_items(chapters: List[Chapter]) -> Dict[str, str]:
    """Figures through the whole thesis, tables within each chapter."""
    labels: Dict[str, str] = {}
    fig = eq_all = tab_all = 0
    by_section = STYLE.get("equations") == "section"
    plain = STYLE.get("equations") == "plain"
    for ch in chapters:
        tab = eq = 0
        sec = f"{ch.number}.0"
        for b in ch.blocks:
            if b.kind == "section" and by_section:
                sec, eq = b.text.split()[0], 0
            if b.kind == "figure":
                fig += 1
                b.number = f"Figure {fig}"
            elif b.kind == "display":
                eq += 1
                eq_all += 1
                if plain:
                    b.number = f"Equation ({eq_all})"
                elif by_section:
                    b.number = f"Eqn {sec}.{eq}"
                else:
                    b.number = f"Equation {ch.number}.{eq}"
            elif b.kind == "table":
                tab += 1
                tab_all += 1
                plain_tab = STYLE.get("tables") == "plain"
                b.number = f"Table {tab_all}" if plain_tab else f"Table {ch.number}.{tab}"
            else:
                continue
            if b.label:
                if b.label in labels:
                    raise SystemExit(f"duplicate label {b.label}")
                labels[b.label] = b.number
    return labels


CITE_RE = re.compile(r"\[(@[\w-]+(?:\s*,\s*@[\w-]+)*)\]")
# Two or more citation groups written next to each other count as one.
CITE_RUN_RE = re.compile(r"\[(@[\w-]+(?:\s*,\s*@[\w-]+)*)\](?:\s*,?\s*\[(@[\w-]+(?:\s*,\s*@[\w-]+)*)\])+")


def format_citation(numbers: List[int]) -> str:
    """[1],[2],[3],[5] -> [1-3],[5]; [1],[2] -> [1-2]; [1],[3],[6] unchanged."""
    nums = sorted(set(numbers))
    out, k = [], 0
    while k < len(nums):
        j = k
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"[{nums[k]}-{nums[j]}]" if j > k else f"[{nums[k]}]")
        k = j + 1
    return ", ".join(out)


def resolve_refs(text: str, labels: Dict[str, str]) -> str:
    def sub(m):
        key = m.group(1)
        if key not in labels:
            raise SystemExit(f"unknown reference {{{key}}}")
        return labels[key]

    def cite(m):
        keys = [k.strip() for k in re.findall(r"@[\w-]+", m.group(0))]
        missing = [k for k in keys if k not in labels]
        if missing:
            raise SystemExit(f"citation key not in references.md: {missing}")
        return format_citation([int(labels[k]) for k in keys])
    text = CITE_RUN_RE.sub(cite, text)
    text = CITE_RE.sub(cite, text)
    return re.sub(r"\{((?:fig|tab|eq):[\w.-]+)\}", sub, text)


def block_texts(b: Block) -> List[str]:
    return [b.text] + list(b.items) + [c for r in b.rows for c in r]


def number_citations(chapters: List[Chapter], labels: Dict[str, str]) -> Chapter:
    """Number citations by first use and build the References chapter."""
    refs: Dict[str, str] = {}
    for line in (CONTENT / "references.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(@[\w-]+):\s*(.+)$", line.strip())
        if m:
            refs[m.group(1)] = m.group(2)
    order: List[str] = []
    for ch in chapters:
        for b in ch.blocks:
            if b.kind == "code":
                continue
            for txt in block_texts(b):
                for m in CITE_RE.finditer(txt or ""):
                    for k in (x.strip() for x in m.group(1).split(",")):
                        if k not in refs:
                            raise SystemExit(f"citation {k} not in references.md")
                        if k not in order:
                            order.append(k)
    for n, k in enumerate(order, 1):
        labels[k] = str(n)
    unused = [k for k in refs if k not in order]
    if unused:
        log(f"note: {len(unused)} reference(s) not cited: {', '.join(unused)}")
    items = [f"[{n}]\t{refs[k]}" for n, k in enumerate(order, 1)]
    return Chapter(chapters[-1].number + 1, "References", [Block("refs", items=items)])


# --------------------------------------------------------------------------
# Writing helpers
# --------------------------------------------------------------------------

INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\*[^*\s][^*]*?\*)")


def add_runs(p, text: str, size: Optional[float] = None, bold: bool = False,
             italic: bool = False, font: str = BODY_FONT) -> None:
    for part in INLINE.split(text):
        if not part:
            continue
        b, it, f = bold, italic, font
        if part.startswith("**") and part.endswith("**"):
            part, b = part[2:-2], True
        elif part.startswith("`") and part.endswith("`"):
            part, f = part[1:-1], CODE_FONT
        elif part.startswith("*") and part.endswith("*"):
            part, it = part[1:-1], True
        r = p.add_run(part)
        r.bold, r.italic = b, it
        r.font.name = f
        r._element.rPr.rFonts.set(qn("w:eastAsia"), f)
        if size:
            r.font.size = Pt(size if f == BODY_FONT else size - 1.5)


def fmt(p, align=None, ls=1.5, before=0, after=0, first=None, left=None,
        keep_next=False, page_break=False, outline: Optional[int] = None):
    pf = p.paragraph_format
    if align is not None:
        p.alignment = align
    if ls == 1.0:
        pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    else:
        pf.line_spacing = ls
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if first is not None:
        pf.first_line_indent = Inches(first)
    if left is not None:
        pf.left_indent = Inches(left)
    pf.keep_with_next = keep_next
    pf.page_break_before = page_break
    if outline is not None:
        el = OxmlElement("w:outlineLvl")
        el.set(qn("w:val"), str(outline))
        p._p.get_or_add_pPr().append(el)
    return p


def shade(cell, fill: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def set_cell(cell, text: str, bold=False, size=10.0, align=WD_ALIGN_PARAGRAPH.LEFT):
    cell.text = ""
    lines = text.split("<br>")
    for k, line in enumerate(lines):
        p = cell.paragraphs[0] if k == 0 else cell.add_paragraph()
        fmt(p, align=align, ls=1.0, before=1, after=1)
        add_runs(p, line, size=size, bold=bold)


def repeat_header(row) -> None:
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trPr.append(el)


def column_widths(rows: List[List[str]], total: float) -> List[float]:
    n = max(len(r) for r in rows)
    lens = [0.0] * n
    for r in rows:
        for j, c in enumerate(r):
            longest = max((len(x) for x in re.sub(r"[*`]", "", c).split("<br>")), default=0)
            lens[j] = max(lens[j], min(longest, 60) + 4)
    # No column narrower than 0.75 in, or than its longest word (about 0.068 in
    # per character at 10 pt, plus cell padding), so words are never broken.
    mins = [0.75] * n
    for r in rows:
        for j, c in enumerate(r):
            words = re.sub(r"[*`]", "", c).replace("<br>", " ").split()
            longest_word = max((len(w) for w in words), default=0)
            mins[j] = max(mins[j], 0.068 * longest_word + 0.16)
    if sum(mins) >= total:
        return [m * total / sum(mins) for m in mins]
    # Share the width in proportion to content, holding any column that would
    # fall below its minimum at that minimum.
    fixed: Dict[int, float] = {}
    while True:
        free = total - sum(fixed.values())
        s = sum(x for j, x in enumerate(lens) if j not in fixed)
        widths = [fixed.get(j, free * lens[j] / s) for j in range(n)]
        low = [j for j in range(n) if j not in fixed and widths[j] < mins[j]]
        if not low:
            return widths
        for j in low:
            fixed[j] = mins[j]


# --------------------------------------------------------------------------
# The document
# --------------------------------------------------------------------------

class Thesis:
    def __init__(self, meta: dict, front: Dict[str, List[Block]],
                 chapters: List[Chapter], labels: Dict[str, str],
                 pages: Optional[Dict[str, str]] = None) -> None:
        self.meta, self.front, self.chapters = meta, front, chapters
        self.labels, self.pages = labels, pages or {}
        self.doc = docx.Document(str(TEMPLATE))
        body = self.doc.element.body
        sects = body.findall(".//" + qn("w:sectPr"))
        self.title_sect = copy.deepcopy(sects[0])      # roman, title page, footer refs
        self.main_sect = body.find(qn("w:sectPr"))     # body-level, starts at page 1
        for child in list(body):
            if child is not self.main_sect:
                body.remove(child)
        mar = self.main_sect.find(qn("w:pgMar"))
        for k, v in (("top", 1440), ("bottom", 1800), ("left", 1701), ("right", 1080),
                     ("footer", 567)):
            mar.set(qn(f"w:{k}"), str(v))

    # -- primitives ---------------------------------------------------------
    def p(self, text="", **kw):
        size = kw.pop("size", None)
        bold = kw.pop("bold", False)
        italic = kw.pop("italic", False)
        style = kw.pop("style", None)
        para = self.doc.add_paragraph(style=style)
        if text:
            add_runs(para, text, size=size, bold=bold, italic=italic)
        return fmt(para, **kw)

    def page_no(self, key: str) -> str:
        return self.pages.get(key, PLACEHOLDER)

    # -- front matter -------------------------------------------------------
    def title_page(self):
        m = self.meta
        C = WD_ALIGN_PARAGRAPH.CENTER
        self.p(m["title"], size=16, bold=True, align=C, ls=1.15, after=24)
        self.p("A CAPSTONE PROJECT REPORT", bold=True, align=C, after=18)
        for line in ("Submitted in partial fulfillment of the",
                     "requirement for the award of the", "Degree of"):
            self.p(line, italic=True, align=C, ls=1.0)
        self.p(m["degree"], bold=True, align=C, ls=1.0, before=18)
        self.p("IN", bold=True, align=C, ls=1.0)
        self.p(m["branch"], bold=True, align=C, ls=1.0, after=18)
        self.p("by", italic=True, align=C, after=6)
        for name, reg in m["students"]:
            self.p(f"{name} ({reg})", size=13, bold=True, align=C, ls=1.15)
        self.p("Under the Guidance of", italic=True, align=C, before=18, after=6)
        self.p(m["guide"], bold=True, align=C, after=18)
        logo = self.p(align=C, after=18)
        logo.add_run().add_picture(str(LOGO), width=Inches(2.6))
        for line in (m["school"], m["university"], m["city"]):
            self.p(line, size=14, align=C, ls=1.0)
        last = self.p(m["month_year"], size=14, italic=True, align=C, before=12)
        last._p.get_or_add_pPr().append(self.title_sect)      # end of section 1

    def certificate(self):
        m = self.meta
        C, J = WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.JUSTIFY
        self.p("CERTIFICATE", size=14, bold=True, align=C, after=18)
        names = ", ".join(f"**{n.title()} ({r})**" for n, r in m["students"])
        text = (f"This is to certify that the Capstone Project work titled "
                f"**“{m['title']}”** that is being submitted by {names} "
                f"is in partial fulfillment of the requirements for the award of "
                f"Bachelor of Technology in {m['branch_title_case']}, is a record of "
                f"bonafide work done under my guidance. The contents of this "
                f"Project work, in full or in parts, have neither been taken from "
                f"any other source nor have been submitted to any other Institute "
                f"or University for award of any degree or diploma and the same "
                f"is certified.")
        self.p(text, size=12, align=J, ls=1.5, after=36)
        R = WD_ALIGN_PARAGRAPH.RIGHT
        self.p(m["guide_signature"], align=R, ls=1.0, before=24)
        self.p("Guide", align=R, ls=1.0, after=36)
        self.p("The thesis is satisfactory / unsatisfactory", bold=True, align=C, after=60)
        self.two_col("Internal Examiner", "External Examiner", bold=True, after=48)
        self.p("Approved by", bold=True, align=C, after=60)
        self.two_col("PROGRAM CHAIR", "DEAN", bold=True)
        self.two_col(m["program_chair"], m["school_title_case"])

    def two_col(self, left, right, bold=False, after=0):
        para = self.p(ls=1.0, after=after)
        para.paragraph_format.tab_stops.add_tab_stop(Inches(6.2), alignment=2)  # right
        add_runs(para, f"{left}\t{right}", bold=bold)

    def front_section(self, title, blocks: List[Block]):
        self.p(title, size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
               after=18, page_break=True)
        for b in blocks:
            self.block(b, chapter=0)

    def contents(self):
        C = WD_ALIGN_PARAGRAPH.CENTER
        self.p("TABLE OF CONTENTS", size=14, bold=True, align=C, after=12, page_break=True)
        rows = [["S.No.", "Chapter", "Title", "Page Number"]]
        n = 0
        for ch in self.chapters:
            n += 1
            chap_no = "" if is_references(ch) and STYLE.get("references") == "unnumbered"                 else str(ch.number)
            nums, titles, pages = [chap_no], [ch.title], [self.page_no(f"ch:{ch.number}")]
            for b in ch.blocks:
                if b.kind == "section" and ch.title.lower() != "appendix":
                    num, _, t = b.text.partition(" ")
                    nums.append(num)
                    titles.append(t)
                    pages.append(self.page_no(f"sec:{b.text}"))
            rows.append([f"{n}.", "<br>".join(nums), "<br>".join(titles), "<br>".join(pages)])
        self.table(rows, widths=[0.6, 0.8, 4.0, 0.9], header_fill="FFFFFF", size=10,
                   keep_together=False)

    def list_of(self, title, kind, key):
        C = WD_ALIGN_PARAGRAPH.CENTER
        self.p(title, size=14, bold=True, align=C, after=12, page_break=True)
        head = "Table No." if kind == "table" else "Figure No."
        rows = [[head, "Title", "Page No."]]
        for ch in self.chapters:
            for b in ch.blocks:
                if b.kind == kind:
                    rows.append([b.number.split()[1], resolve_refs(b.text, self.labels),
                                 self.page_no(f"{kind}:{b.number}")])
        self.table(rows, widths=[1.0, 4.4, 0.9], header_fill="FFFFFF", size=10,
                   center_cols=(0, 2), keep_together=False)

    # -- body ---------------------------------------------------------------
    def chapter(self, ch: Chapter):
        C = WD_ALIGN_PARAGRAPH.CENTER
        # The References list is headed by its title alone, without "CHAPTER n".
        numbered = not is_references(ch)
        if numbered:
            self.p(f"CHAPTER {ch.number}", size=14, bold=True, align=C, page_break=True,
                   keep_next=True)
        self.p(ch.title.upper(), size=14, bold=True, align=C, after=18, keep_next=True,
               outline=0, page_break=not numbered)
        for b in ch.blocks:
            self.block(b, chapter=ch.number)

    def block(self, b: Block, chapter: int):
        J, C = WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.CENTER
        text = resolve_refs(b.text, self.labels) if b.text else ""
        if b.kind == "section":
            self.p(text, size=14, bold=True, before=12, after=6, keep_next=True, outline=1)
        elif b.kind == "subsection":
            self.p(text, size=13, bold=True, before=10, after=4, keep_next=True, outline=2)
        elif b.kind == "minor":
            self.p(text, size=12, bold=True, before=8, after=2, keep_next=True)
        elif b.kind == "para":
            # A sentence that introduces a list, equation or listing stays on
            # the same page as what it introduces.
            lead_in = text.rstrip().endswith((":", "as follows.", "is given by", "follows:"))
            self.p(text, align=J, first=0.5, after=6, keep_next=lead_in)
        elif b.kind in ("bullets", "numbers"):
            for k, item in enumerate(b.items, 1):
                mark = "•" if b.kind == "bullets" else f"{k}."
                para = self.p(style="List Paragraph", align=J, left=0.75, first=-0.25,
                              after=3, ls=1.5)
                para.paragraph_format.tab_stops.add_tab_stop(Inches(0.75))
                add_runs(para, f"{mark}\t" + resolve_refs(item, self.labels))
        elif b.kind == "display":
            # Equation centred, its number at the right margin.
            para = self.p(before=4, after=8, ls=1.0)
            stops = para.paragraph_format.tab_stops
            stops.add_tab_stop(Inches(TEXT_WIDTH_IN / 2), WD_TAB_ALIGNMENT.CENTER)
            stops.add_tab_stop(Inches(TEXT_WIDTH_IN), WD_TAB_ALIGNMENT.RIGHT)
            add_runs(para, "	", size=12)
            add_runs(para, text, size=12, italic=True)
            last = b.number.split()[-1]
            num = b.number if b.number.startswith("Eqn") else (
                last if last.startswith("(") else f"({last})")
            add_runs(para, "	" + num, size=12)
        elif b.kind == "code":
            code_lines = b.text.split("\n")
            short = len(code_lines) <= 25          # keep short listings on one page
            for k, line in enumerate(code_lines):
                para = self.doc.add_paragraph(style="Plain Text")
                fmt(para, ls=1.0, left=0.25, before=0, after=0,
                    keep_next=short and k < len(code_lines) - 1)
                r = para.add_run(line if line else " ")
                r.font.name = CODE_FONT
                r.font.size = Pt(8.5)
            self.p(after=4, ls=1.0)
        elif b.kind == "figure":
            path = (ROOT / b.path) if not Path(b.path).is_absolute() else Path(b.path)
            if b.landscape:
                self.orientation(landscape=True)
            para = self.p(align=C, before=6, after=2, keep_next=True, ls=1.0)
            max_w = LANDSCAPE_WIDTH_IN if b.landscape else TEXT_WIDTH_IN
            if path.exists():
                para.add_run().add_picture(str(path), width=Inches(min(b.width, max_w)))
            else:
                # Not made yet (see PROGRESS.md); keep the build going and show it.
                log(f"WARNING: figure not found, placeholder used: {b.path}")
                add_runs(para, f"[ figure not yet generated: {b.path} ]", size=11, bold=True)
            cap = self.p(style="Caption", align=C, after=12, ls=1.0)
            add_runs(cap, f"{b.number} {text}", size=10, bold=True)
            if b.landscape:
                self.orientation(landscape=False)
        elif b.kind == "table":
            cap = self.p(align=C, before=8, after=4, keep_next=True, ls=1.0)
            add_runs(cap, f"{b.number} {text}", size=11, bold=True)
            rows = [[resolve_refs(c, self.labels) for c in r] for r in b.rows]
            self.table(rows, all_left=STYLE.get("table_align") == "left")
            self.p(after=4, ls=1.0)
        elif b.kind == "refs":
            for item in b.items:
                # "[n]<tab>" so the first line's text aligns with the hanging indent.
                para = self.p(item, size=11, align=WD_ALIGN_PARAGRAPH.JUSTIFY, ls=1.0,
                              left=0.45, first=-0.45, after=2)
                para.paragraph_format.tab_stops.add_tab_stop(Inches(0.45))
        elif b.kind == "pagebreak":
            self.p(page_break=True, ls=1.0)

    def orientation(self, landscape: bool) -> None:
        """Start a new section on a new page, landscape or back to portrait.

        The new section inherits the footer, and its page numbers continue
        from the previous section instead of restarting.
        """
        sect = self.doc.add_section(WD_SECTION.NEW_PAGE)
        pg = sect._sectPr.find(qn("w:pgNumType"))
        if pg is not None and pg.get(qn("w:start")) is not None:
            del pg.attrib[qn("w:start")]
        w, h = sect.page_width, sect.page_height
        if landscape != (w > h):
            sect.page_width, sect.page_height = h, w
        sect.orientation = WD_ORIENT.LANDSCAPE if landscape else WD_ORIENT.PORTRAIT

    def table(self, rows, widths=None, header_fill="E7E6E6", size=10.0, center_cols=(),
              keep_together=True, all_left=False):
        n = max(len(r) for r in rows)
        rows = [r + [""] * (n - len(r)) for r in rows]
        widths = widths or column_widths(rows, TEXT_WIDTH_IN)
        body = rows[1:]
        numeric_col = [j in center_cols or (bool(body) and sum(
            is_numeric(r[j]) for r in body) * 2 > len(body)) for j in range(n)]
        t = self.doc.add_table(rows=len(rows), cols=n)
        t.style = self.doc.styles["Table Grid"]
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        for i, r in enumerate(rows):
            for j, c in enumerate(r):
                cell = t.cell(i, j)
                cell.width = Inches(widths[j])
                # One alignment per column: centred if most of it is numbers.
                centred = numeric_col[j] and not all_left
                align = WD_ALIGN_PARAGRAPH.CENTER if centred else WD_ALIGN_PARAGRAPH.LEFT
                set_cell(cell, capitalise(c), bold=(i == 0), size=size, align=align)
                if i == 0 and header_fill:
                    shade(cell, header_fill)
        repeat_header(t.rows[0])
        for row in t.rows:
            trPr = row._tr.get_or_add_trPr()
            el = OxmlElement("w:cantSplit")
            el.set(qn("w:val"), "true")
            trPr.append(el)
        if keep_together:
            # The whole table on one page: every row but the last keeps with the next.
            for row in t.rows[:-1]:
                for cell in row.cells:
                    for para in cell.paragraphs:
                        para.paragraph_format.keep_with_next = True
        return t

    # -- assemble -----------------------------------------------------------
    def build(self) -> docx.Document:
        self.title_page()
        self.certificate()
        self.front_section("ACKNOWLEDGEMENTS", self.front.get("acknowledgements", []))
        self.front_section("ABSTRACT", self.front.get("abstract", []))
        self.contents()
        self.list_of("List of Tables", "table", "lot")
        self.list_of("List of Figures", "figure", "lof")
        self.start_body_numbering()
        for ch in self.chapters:
            self.chapter(ch)
        self.drop_unused_media()
        return self.doc

    def start_body_numbering(self) -> None:
        """Close the front matter, which carries no page number, and start the body at 1."""
        self.doc.add_section(WD_SECTION.NEW_PAGE)
        front, body = self.doc.sections[-2], self.doc.sections[-1]
        # The body keeps the template's numbered footer; the front matter gets an empty one.
        numbered = [copy.deepcopy(p._p) for p in front.footer.paragraphs]
        body.footer.is_linked_to_previous = False
        ftr = body.footer._element
        for p in ftr.findall(qn("w:p")):
            ftr.remove(p)
        for p in numbered:
            ftr.append(p)
        front.footer.is_linked_to_previous = False
        pg = body._sectPr.find(qn("w:pgNumType"))
        if pg is None:
            pg = OxmlElement("w:pgNumType")
            body._sectPr.append(pg)
        pg.set(qn("w:fmt"), "decimal")
        pg.set(qn("w:start"), "1")

    def drop_unused_media(self):
        """Remove the template's own images and chart, which nothing references now."""
        xml = self.doc.element.xml
        part = self.doc.part
        for rId, rel in list(part.rels.items()):
            if rel.reltype in (RT.IMAGE, RT.CHART) and f'"{rId}"' not in xml:
                del part.rels[rId]


def load_front() -> Dict[str, List[Block]]:
    path = CONTENT / "00_front.md"
    out: Dict[str, List[Block]] = {}
    if not path.exists():
        return out
    current = None
    buf: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines() + ["@end"]:
        if line.startswith("@"):
            if current:
                out[current] = parse_markdown("\n".join(buf))
            current, buf = line[1:].strip().lower(), []
        else:
            buf.append(line)
    return out


# --------------------------------------------------------------------------
# Page numbers, via Word's own rendering
# --------------------------------------------------------------------------

def export_pdf(src: Path, dst: Path) -> None:
    ps = (f"$w = New-Object -ComObject Word.Application; $w.Visible = $false; "
          f"$d = $w.Documents.Open('{src}', $false, $true); "
          f"$d.SaveAs([ref]'{dst}', [ref]17); $d.Close($false); $w.Quit()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


def pdf_pages(pdf: Path) -> List[str]:
    out = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], check=True,
                         capture_output=True).stdout.decode("utf-8", "replace")
    return [re.sub(r"\s+", " ", pg) for pg in out.split("\f")]


def is_references(ch: Chapter) -> bool:
    return ch.title == "References"


def printed_number(page_text: str) -> Optional[str]:
    m = re.search(r"(?:^|\s)(\d+|[ivxlc]+)\s*$", page_text.strip())
    return m.group(1) if m else None


def locate(chapters: List[Chapter], pages: List[str]) -> Dict[str, str]:
    """Find the printed page of every chapter, section, table and figure."""
    targets: List[Tuple[str, str]] = []
    for ch in chapters:
        targets.append((f"ch:{ch.number}", ch.title.upper() if is_references(ch)
                        else f"CHAPTER {ch.number} {ch.title.upper()}"))
        for b in ch.blocks:
            if b.kind == "section":
                targets.append((f"sec:{b.text}", b.text))
            elif b.kind in ("table", "figure"):
                cap = re.sub(r"[*`]", "", b.text)
                targets.append((f"{b.kind}:{b.number}", f"{b.number} {cap[:25]}"))
    found: Dict[str, str] = {}
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    # The body starts at the first chapter heading after the front matter.
    start = next(i for i, pg in enumerate(pages)
                 if norm(targets[0][1]) in pg and "TABLE OF CONTENTS" not in pg
                 and "List of" not in pg)
    cur = start
    for key, needle in targets:
        needle = norm(needle)
        k = cur
        while k < len(pages) and needle not in pages[k]:
            k += 1
        if k == len(pages):
            log(f"WARNING: could not place {needle!r}")
            continue
        cur = k
        found[key] = printed_number(pages[k]) or str(k)
    for key, needle in (("front:lot", "List of Tables"), ("front:lof", "List of Figures")):
        for k, pg in enumerate(pages[:start]):
            if "TABLE OF CONTENTS" in pg:
                continue                  # the contents page lists both titles
            if pg.strip().startswith(needle) or f" {needle} " in pg[:80]:
                found[key] = printed_number(pg) or str(k)
                break
    return found


# --------------------------------------------------------------------------
# Abbreviations and the writing rules of the v2 document
# --------------------------------------------------------------------------

PROSE = ("para", "bullets", "numbers")
# Spans that are never rewritten: code, citations and cross-references.
_PROTECTED = re.compile(r"`[^`]+`|\[@[^\]]+\]|\{(?:fig|tab|eq):[\w.-]+\}")


def load_abbreviations(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([^#:\s][^:]*?)\s*:\s*(.+)$", line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def abbr_pattern(abbr: str) -> re.Pattern:
    # Whole word only, and not part of a hyphenated name such as DP-FedFace.
    return re.compile(rf"(?<![\w-]){re.escape(abbr)}(s?)(?![\w-])")


def doc_order(front: Dict[str, List[Block]], chapters: List[Chapter]):
    for key in ("acknowledgements", "abstract"):
        for b in front.get(key, []):
            yield b
    for ch in chapters:
        for b in ch.blocks:
            yield b


def _texts(b: Block) -> List[Tuple[str, int]]:
    """(text, index) pairs of a block; index -1 is b.text, k is b.items[k]."""
    if b.kind in ("bullets", "numbers"):
        return [(x, k) for k, x in enumerate(b.items)]
    return [(b.text, -1)]


def _set(b: Block, k: int, text: str) -> None:
    if k < 0:
        b.text = text
    else:
        b.items[k] = text


def expand_abbreviations(front, chapters, path: Path) -> None:
    """Write each abbreviation out in full at its first use in running text."""
    glossary = load_abbreviations(path)
    defined: Dict[str, bool] = {a: False for a in glossary}
    early: Dict[str, str] = {}
    for b in doc_order(front, chapters):
        if b.kind == "code" or b.kind == "display":
            continue
        for text, k in _texts(b):
            if not text:
                continue
            for abbr, full in glossary.items():
                if defined[abbr]:
                    continue
                if f"({abbr})" in text or f"({abbr}s)" in text:
                    defined[abbr] = True            # written out by hand
                    continue
                pat = abbr_pattern(abbr)
                hits = [m for m in pat.finditer(text)
                        if not any(s.start() <= m.start() < s.end()
                                   for s in _PROTECTED.finditer(text))]
                if not hits:
                    continue
                if b.kind not in PROSE:
                    early.setdefault(abbr, f"{b.kind}: {text[:60]}")
                    continue
                m = hits[0]
                plural = m.group(1)
                long = full + ("s" if plural and not full.endswith("s") else "")
                text = text[:m.start()] + f"{long} ({abbr}{plural})" + text[m.end():]
                _set(b, k, text)
                defined[abbr] = True
    for abbr, where in early.items():
        if not defined[abbr] or True:
            log(f"lint: {abbr} appears in a {where!r} before it is defined in running text")
    unused = [a for a, d in defined.items() if not d]
    if unused:
        log(f"note: abbreviations never used: {', '.join(unused)}")
    STYLE["_glossary"] = glossary


def lint(front, chapters) -> None:
    """Warnings for the v2 writing rules. They never stop the build."""
    glossary: Dict[str, str] = STYLE.get("_glossary", {})       # type: ignore
    seen_full: Dict[str, bool] = {}
    blocks = list(doc_order(front, chapters))
    n_warn = 0

    def warn(msg):
        nonlocal n_warn
        n_warn += 1
        log(f"lint: {msg}")

    lines = lambda s: len(re.sub(r"\[@[^\]]+\]|\{[^}]+\}|[*`]", "", s)) / 95
    for i, b in enumerate(blocks):
        nxt = blocks[i + 1] if i + 1 < len(blocks) else None
        if b.kind in ("bullets", "numbers"):
            warn(f"list instead of prose: {b.items[0][:60]!r}")
        if b.kind == "display" and (nxt is None or nxt.kind != "para"):
            warn(f"equation without an explanation paragraph: {b.text[:40]!r}")
        if b.kind in ("section", "subsection") and nxt is not None \
                and nxt.kind in ("section", "subsection", "minor"):
            warn(f"heading followed directly by a heading: {b.text!r}")
        if b.kind == "section" and nxt is not None and nxt.kind == "para":
            # A section with sub-sections opens with a paragraph of 5-6 lines.
            following = next((x for x in blocks[i + 1:]
                              if x.kind in ("section", "subsection")), None)
            if following is not None and following.kind == "subsection" \
                    and lines(nxt.text) < 4.5:
                warn(f"lead paragraph under {b.text!r} is {lines(nxt.text):.1f} lines")
        if b.kind not in PROSE:
            continue
        for text, _ in _texts(b):
            clean = _PROTECTED.sub("", text)
            for m in re.finditer(r"\(([^)]*)\)", clean):
                inner = m.group(1).rstrip("s")
                if inner not in glossary:
                    warn(f"bracket in prose: {clean[max(0, m.start() - 30):m.end() + 10]!r}")
            if "?" in clean:
                warn(f"question in prose: {clean[:70]!r}")
            for m in re.finditer(r"\[@[^\]]+\](?:\s*,?\s*\[@[^\]]+\])*", text):
                after = text[m.end():m.end() + 2]
                if after and not re.match(r"^[.\s]*$", after[:1]) and after[:1] != ".":
                    warn(f"citation not at the end of a sentence: "
                         f"{text[max(0, m.start() - 40):m.end() + 15]!r}")
            # A definition "Full Form (ABBR)" is not a later use of its own words,
            # nor of an abbreviation inside it, such as CPU inside vCPU.
            plain = clean
            for ab, fu in glossary.items():
                plain = re.sub(re.escape(f"{fu} ({ab}") + r"s?\)", " ", plain)
            for abbr, full in glossary.items():
                if f"{full} ({abbr}" in clean:
                    seen_full[abbr] = True
                    continue
                if re.search(re.escape(full), plain, re.I):
                    if seen_full.get(abbr):
                        warn(f"full form after definition, use {abbr}: {full!r}")
                    seen_full[abbr] = True
                elif abbr_pattern(abbr).search(clean):
                    seen_full[abbr] = True
        if b.kind == "para" and nxt is not None and nxt.kind == "para" \
                and lines(b.text) < 4 and lines(nxt.text) < 4:
            warn(f"two short paragraphs in a row: {b.text[:60]!r}")
    for ch in chapters:
        if ch.blocks and ch.blocks[0].kind == "para" and lines(ch.blocks[0].text) < 4.5:
            warn(f"chapter {ch.number} opens with a {lines(ch.blocks[0].text):.1f}-line paragraph")
    log(f"lint: {n_warn} warning(s)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true", help="skip Word rendering")
    ap.add_argument("--content", default="content",
                    help="content folder under docs/thesis, e.g. content_v2")
    ap.add_argument("--out", default=None,
                    help="output name in docs/, e.g. FedMPS_Capstone_Thesis_v2")
    args = ap.parse_args()

    global CONTENT, OUT_DOCX, OUT_PDF
    CONTENT = HERE / args.content
    if args.out:
        OUT_DOCX = ROOT / "docs" / f"{args.out}.docx"
        OUT_PDF = OUT_DOCX.with_suffix(".pdf")

    meta = json.loads((CONTENT / "meta.json").read_text(encoding="utf-8"))
    STYLE.update(meta.get("style", {}))
    front = load_front()
    chapters = load_chapters()
    labels = number_items(chapters)
    chapters.append(number_citations(chapters, labels))
    if STYLE.get("abbreviations"):
        expand_abbreviations(front, chapters, CONTENT / str(STYLE["abbreviations"]))
    if STYLE.get("lint"):
        lint(front, chapters)
    log(f"{len(chapters)} chapters, "
        f"{sum(b.kind == 'figure' for c in chapters for b in c.blocks)} figures, "
        f"{sum(b.kind == 'table' for c in chapters for b in c.blocks)} tables")

    Thesis(meta, front, chapters, labels).build().save(str(OUT_DOCX))
    if args.no_pdf:
        log(f"wrote {OUT_DOCX} (page numbers not filled)")
        return 0

    pages: Dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for attempt in range(3):
            pdf = Path(tmp) / f"pass{attempt}.pdf"
            export_pdf(OUT_DOCX, pdf)
            new = locate(chapters, pdf_pages(pdf))
            if new == pages:
                break
            pages = new
            Thesis(meta, front, chapters, labels, pages).build().save(str(OUT_DOCX))
    export_pdf(OUT_DOCX, OUT_PDF)
    n = len(pdf_pages(OUT_PDF)) - 1
    log(f"wrote {OUT_DOCX.name} and {OUT_PDF.name} ({n} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
