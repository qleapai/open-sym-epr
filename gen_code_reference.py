"""Generate a complete function-by-function code-reference DOCX for EPR-Suite.

Walks every Python module in the combined codebase, extracts the module
docstring, every class and function with its signature, docstring (the embedded
logic/comments), and full source, and renders them into a formatted Word
document. Run with: python gen_code_reference.py
"""
from __future__ import annotations
import ast, os
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path(r"E:\Open-Sym-EPR")
OUT = ROOT / "paper" / "OpenSymEPR_Code_Reference.docx"
NAVY = RGBColor(0x16, 0x30, 0x6b)
TEAL = RGBColor(0x2c, 0x3e, 0x50)
CODEC = RGBColor(0x1a, 0x1a, 0x1a)

# ordered groups of modules
GROUPS = [
    ("Part I — Open-Sym-EPR engine (native-Python spin-Hamiltonian solvers)", [
        "openspin/__init__.py", "openspin/constants.py", "openspin/spin_operators.py",
        "openspin/spin_hamiltonian.py", "openspin/powder.py", "openspin/lineshapes.py",
        "openspin/cw.py", "openspin/endor.py", "openspin/eseem.py", "openspin/magnetometry.py",
        "openspin/fitting.py", "openspin/io.py", "openspin/bruker.py", "openspin/science.py",
    ]),
    ("Part II — Open-Sym-EPR engine (simulation, fitting, batch, export)", [
        "epr_simfit/__init__.py", "epr_simfit/constants.py", "epr_simfit/spin_models.py",
        "epr_simfit/lineshapes.py", "epr_simfit/simulator.py", "epr_simfit/spin_operators.py",
        "epr_simfit/spin_hamiltonian.py", "epr_simfit/powder.py", "epr_simfit/preprocessing.py",
        "epr_simfit/fitter.py", "epr_simfit/model_library.py", "epr_simfit/model_comparison.py",
        "epr_simfit/model_suggester.py", "epr_simfit/interpretation.py", "epr_simfit/io.py",
        "epr_simfit/bruker.py", "epr_simfit/metadata_parser.py", "epr_simfit/batch.py",
        "epr_simfit/export.py", "epr_simfit/reference_library.py", "epr_simfit/user_models.py",
        "epr_simfit/native_solvers.py", "epr_simfit/ml_fitting.py", "epr_simfit/plotting.py", "epr_simfit/report.py",
        "epr_simfit/evidence.py", "epr_simfit/demo_data.py", "epr_simfit/utils.py",
        "epr_simfit/about.py", "epr_simfit/cli.py",
    ]),
    ("Part III — Application interfaces (Streamlit)", ["app_openspin.py", "app_simepr.py"]),
    ("Part IV — Tests", ["tests/test_openspin.py", "tests/test_all_capabilities.py",
                         "tests/test_native_solvers.py"]),
]

doc = Document()
# base style
normal = doc.styles["Normal"]; normal.font.name = "Calibri"; normal.font.size = Pt(10.5)
for lvl, sz in [("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 11.5)]:
    s = doc.styles[lvl]; s.font.size = Pt(sz); s.font.color.rgb = NAVY if lvl == "Heading 1" else TEAL


def shade(par, fill):
    pPr = par._p.get_or_add_pPr()
    sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:fill"), fill)
    pPr.append(sh)


def code_block(text, size=7.6):
    """Add a shaded monospace block preserving all lines (full source)."""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.12)
    p.paragraph_format.space_before = Pt(2); p.paragraph_format.space_after = Pt(6)
    shade(p, "F4F4F4")
    lines = text.split("\n")
    for k, ln in enumerate(lines):
        r = p.add_run(("" if k == 0 else "\n") + ln.rstrip("\n"))
        r.font.name = "Consolas"; r.font.size = Pt(size); r.font.color.rgb = CODEC
        # ensure monospace east-asian too
        rpr = r._element.get_or_add_rPr(); rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts"); rpr.append(rf)
        rf.set(qn("w:ascii"), "Consolas"); rf.set(qn("w:hAnsi"), "Consolas")
    return p


def sig_of(node, source):
    """Return a one-line def signature string."""
    seg = ast.get_source_segment(source, node) or ""
    # take up to the first ':' that ends the signature line(s)
    head = []
    depth = 0
    for ch in seg:
        head.append(ch)
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == ":" and depth == 0:
            break
    return "".join(head).strip()


def doc_function(node, source, kind="def"):
    name = node.name
    sig = sig_of(node, source)
    h = doc.add_paragraph(style="Heading 3")
    rr = h.add_run(("class " if kind == "class" else "") + name)
    rr.font.name = "Consolas"
    # signature
    pc = doc.add_paragraph(); rs = pc.add_run(sig); rs.font.name = "Consolas"; rs.font.size = Pt(8.5); rs.bold = True
    # docstring (logic/comments)
    ds = ast.get_docstring(node)
    if ds:
        pd = doc.add_paragraph(); pd.paragraph_format.left_indent = Inches(0.1)
        rd = pd.add_run(ds.strip()); rd.italic = True; rd.font.size = Pt(9.5)
    # full source
    src = ast.get_source_segment(source, node) or ""
    code_block(src)


def doc_module(relpath):
    path = ROOT / relpath
    if not path.exists():
        return
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return
    doc.add_paragraph(style="Heading 2").add_run(relpath).font.name = "Consolas"
    md = ast.get_docstring(tree)
    if md:
        pm = doc.add_paragraph(); rm = pm.add_run("Module purpose: " + md.strip()); rm.italic = True; rm.font.size = Pt(10)

    is_app = relpath.startswith("app_")
    funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]

    if is_app:
        # apps are UI scripts: document helper functions, then include the full script
        for fn in funcs:
            doc_function(fn, source)
        doc.add_paragraph().add_run("Full application script (Streamlit UI):").bold = True
        code_block(source, size=6.6)
        return

    for cl in classes:
        doc_function(cl, source, kind="class")
        # methods
        for m in [n for n in cl.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            mp = doc.add_paragraph(); mr = mp.add_run("method  " + cl.name + "." + m.name); mr.font.name = "Consolas"; mr.bold = True; mr.font.size = Pt(9)
            mds = ast.get_docstring(m)
            if mds:
                md2 = doc.add_paragraph(); rmd = md2.add_run(mds.strip()); rmd.italic = True; rmd.font.size = Pt(9)
            code_block(ast.get_source_segment(source, m) or "")
    for fn in funcs:
        doc_function(fn, source)
    # module-level constants of interest (assignments to UPPER_CASE names)
    consts = []
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id.isupper():
                    consts.append(ast.get_source_segment(source, n) or "")
    if consts:
        doc.add_paragraph().add_run("Module-level constants:").bold = True
        code_block("\n".join(consts))


# ── Title page ──
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
rt = t.add_run("EPR-Suite — Complete Code Reference"); rt.bold = True; rt.font.size = Pt(24); rt.font.color.rgb = NAVY
st = doc.add_paragraph(); st.alignment = WD_ALIGN_PARAGRAPH.CENTER
rst = st.add_run("Open-Sym-EPR + Open-Sym-EPR: every module, class, function, docstring, and source"); rst.font.size = Pt(13); rst.font.color.rgb = TEAL
au = doc.add_paragraph(); au.alignment = WD_ALIGN_PARAGRAPH.CENTER
au.add_run("Md Sakib Hasan Khan — The Australian National University").italic = True
doc.add_paragraph()

# ── Overview ──
doc.add_paragraph(style="Heading 1").add_run("Overview")
ov = ("This document is a complete, automatically generated reference for the EPR-Suite codebase, "
      "which combines two applications: Open-Sym-EPR, a native-Python spin-Hamiltonian engine for cw-EPR, "
      "ENDOR, ESEEM, and magnetometry; and Open-Sym-EPR, a cw-EPR simulation, fitting, batch-processing, and "
      "reporting application. For every module it reproduces the module purpose (its docstring), and for "
      "every class, method, and function it reproduces the signature, the docstring that documents the "
      "logic, and the complete source code. The reference is generated directly from the source by the "
      "script gen_code_reference.py, so it is always exactly synchronized with the code.")
doc.add_paragraph(ov)

# stats
nmods = sum(len(g[1]) for g in GROUPS)
total_lines = 0
for _, mods in GROUPS:
    for m in mods:
        p = ROOT / m
        if p.exists():
            total_lines += len(p.read_text(encoding="utf-8").splitlines())
doc.add_paragraph(f"Modules documented: {nmods}.  Total source lines: {total_lines:,}.")

for title, mods in GROUPS:
    doc.add_page_break()
    doc.add_paragraph(style="Heading 1").add_run(title)
    for m in mods:
        doc_module(m)

doc.save(OUT)
print("Code reference written:", OUT, "| modules:", nmods, "| lines:", total_lines)
