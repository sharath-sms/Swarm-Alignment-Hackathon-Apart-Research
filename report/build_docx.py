"""Build the Apart submission .docx from the filled-in markdown template.

    python report/build_docx.py

pandoc renders the markdown using the original Apart template as reference doc; this script then restyles
the output to match the template (Old Standard TT throughout, boxed title/author/abstract block, compact
headings, booktabs-style tables).
"""
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "Apart Research hackathon submission template.md"
OUT = HERE / "Apart Research hackathon submission template.docx"
REF = HERE / "_template_original" / "Apart Research hackathon submission template.docx"
FONT = "Old Standard TT"


def meta(key: str) -> str:
    m = re.search(rf'^{key}:\s*"(.*)"\s*$', SRC.read_text(), re.M)
    return m.group(1) if m else ""


def rfonts() -> str:
    return f'<w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}" w:eastAsia="{FONT}" w:cs="{FONT}"/>'


def _pstyle(sid, name, ppr, rpr="", based="Normal"):
    return (f'<w:style w:type="paragraph" w:customStyle="1" w:styleId="{sid}"><w:name w:val="{name}"/>'
            f'<w:basedOn w:val="{based}"/><w:qFormat/><w:pPr>{ppr}</w:pPr><w:rPr>{rpr}</w:rPr></w:style>')


_BODY = '<w:spacing w:before="0" w:after="120" w:line="252" w:lineRule="auto"/><w:jc w:val="both"/>'
_CAP = '<w:sz w:val="19"/><w:szCs w:val="19"/>'
_RULE = 'w:val="single" w:sz="8" w:space="0" w:color="000000"'
STYLES = {
    "BodyText": _pstyle("BodyText", "Body Text", _BODY),
    "FirstParagraph": _pstyle("FirstParagraph", "First Paragraph", _BODY, based="BodyText"),
    "Compact": _pstyle("Compact", "Compact", '<w:spacing w:before="20" w:after="20" w:line="240" w:lineRule="auto"/>'),
    "Abstract": _pstyle("Abstract", "Abstract", _BODY),
    "CaptionedFigure": _pstyle("CaptionedFigure", "Captioned Figure",
                               '<w:keepNext/><w:spacing w:before="160" w:after="40"/><w:jc w:val="center"/>'),
    "ImageCaption": _pstyle("ImageCaption", "Image Caption",
                            '<w:spacing w:before="0" w:after="200" w:line="240" w:lineRule="auto"/><w:jc w:val="both"/>', _CAP),
    "TableCaption": _pstyle("TableCaption", "Table Caption",
                            '<w:keepNext/><w:spacing w:before="160" w:after="80" w:line="240" w:lineRule="auto"/>'
                            '<w:jc w:val="both"/>', _CAP),
    "Hyperlink": ('<w:style w:type="character" w:styleId="Hyperlink"><w:name w:val="Hyperlink"/>'
                  '<w:rPr><w:color w:val="1F4E99"/></w:rPr></w:style>'),
    # Booktabs-style table: heavy top/bottom rules, light rule under the header row.
    "Table": ('<w:style w:type="table" w:styleId="Table"><w:name w:val="Table"/><w:basedOn w:val="TableNormal"/>'
              '<w:pPr><w:jc w:val="center"/></w:pPr><w:rPr><w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr>'
              '<w:tblPr><w:jc w:val="center"/><w:tblBorders>'
              '<w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
              '<w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/></w:tblBorders>'
              '<w:tblCellMar><w:left w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar></w:tblPr>'
              f'<w:tblStylePr w:type="firstRow"><w:rPr><w:b/><w:bCs/></w:rPr><w:tcPr><w:tcBorders><w:bottom {_RULE}/>'
              '</w:tcBorders></w:tcPr></w:tblStylePr></w:style>'),
}


def para(text: str, *, bold=False, size=22, jc="center", after=120) -> str:
    b = "<w:b/><w:bCs/>" if bold else ""
    return (f'<w:p><w:pPr><w:spacing w:after="{after}" w:line="240" w:lineRule="auto"/><w:jc w:val="{jc}"/></w:pPr>'
            f'<w:r><w:rPr>{b}<w:sz w:val="{size}"/><w:szCs w:val="{size}"/></w:rPr>'
            f'<w:t xml:space="preserve">{text}</w:t></w:r></w:p>')


def restyle_styles(xml: str) -> str:
    xml = re.sub(r"<w:rFonts [^>]*/>", "", xml)  # drop all theme/explicit fonts, then set one default
    xml = re.sub(r"<w:rPrDefault>.*?</w:rPrDefault>",
                 f'<w:rPrDefault><w:rPr>{rfonts()}<w:sz w:val="22"/><w:szCs w:val="22"/>'
                 f'<w:lang w:val="en-US"/></w:rPr></w:rPrDefault>', xml, flags=re.S)
    sizes = {"Heading1": (28, True, False), "Heading2": (24, True, False), "Heading3": (22, True, True)}
    for sid, (sz, bold, ital) in sizes.items():
        def fix(m, sz=sz, bold=bold, ital=ital):
            s = re.sub(r"<w:rPr>.*?</w:rPr>", "", m.group(0), flags=re.S)
            s = re.sub(r'<w:spacing [^>]*/>', '<w:spacing w:before="240" w:after="80"/>', s)
            rpr = ("<w:b/><w:bCs/>" if bold else "") + ("<w:i/>" if ital else "")
            rpr += f'<w:color w:val="000000"/><w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>'
            return s.replace("</w:style>", f"<w:rPr>{rpr}</w:rPr></w:style>")
        xml = re.sub(rf'<w:style [^>]*w:styleId="{sid}".*?</w:style>', fix, xml, flags=re.S)
    # Styles pandoc references but the Apart template does not define (Word would fall back to Normal).
    missing = "".join(STYLES[sid] for sid in STYLES if f'w:styleId="{sid}"' not in xml)
    xml = xml.replace("</w:styles>", missing + "</w:styles>")
    return xml


def box_header(doc: str) -> str:
    """Wrap Title + author + Abstract paragraphs in a single-cell box with a thick top rule, like the template."""
    body_start = doc.index("<w:body>") + len("<w:body>")
    paras = list(re.finditer(r"<w:p>.*?</w:p>|<w:p [^>]*>.*?</w:p>", doc[body_start:], flags=re.S))
    header, end = [], body_start
    for m in paras:
        style = re.search(r'<w:pStyle w:val="([^"]+)"', m.group(0))
        if not style or style.group(1) not in ("Title", "Author", "AbstractTitle", "Abstract"):
            break
        header.append((style.group(1), m.group(0)))
        end = body_start + m.end()
    if not header:
        raise SystemExit("title block not found in pandoc output")

    cell = []
    for style, p in header:
        if style == "Author":
            cell.append(para(meta("author"), size=24, after=40))
            cell.append(para(meta("affiliation"), size=22, after=160))
            cell.append('<w:p><w:pPr><w:spacing w:after="240"/><w:jc w:val="center"/></w:pPr>'
                        '<w:r><w:rPr><w:b/><w:bCs/></w:rPr><w:t xml:space="preserve">With </w:t></w:r>'
                        f'<w:r><w:t>{meta("with")}</w:t></w:r></w:p>')
        elif style == "AbstractTitle":
            cell.append(para("Abstract", bold=True, size=22, jc="left", after=80))
        elif style == "Abstract":
            p = re.sub(r'<w:pStyle w:val="Abstract"/>',
                       '<w:pStyle w:val="Abstract"/><w:spacing w:after="160" w:line="252" w:lineRule="auto"/>'
                       '<w:jc w:val="both"/>', p)
            cell.append(p)
        else:  # Title
            cell.append(re.sub(r'<w:pStyle w:val="Title"/>',
                               '<w:pStyle w:val="Title"/><w:spacing w:before="240" w:after="240"/>', p))
    tbl = ('<w:tbl><w:tblPr><w:tblW w:w="9360" w:type="dxa"/><w:jc w:val="center"/><w:tblBorders>'
           '<w:top w:val="single" w:sz="24" w:color="000000"/><w:bottom w:val="single" w:sz="8" w:color="000000"/>'
           '</w:tblBorders><w:tblLayout w:type="fixed"/>'
           '<w:tblCellMar><w:left w:w="0" w:type="dxa"/><w:right w:w="0" w:type="dxa"/></w:tblCellMar></w:tblPr>'
           '<w:tblGrid><w:gridCol w:w="9360"/></w:tblGrid><w:tr><w:tc><w:tcPr><w:tcW w:w="9360" w:type="dxa"/>'
           '</w:tcPr>' + "".join(cell) + "</w:tc></w:tr></w:tbl>" + para("", size=4, after=0))
    return doc[:body_start] + tbl + doc[end:]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.docx"
        subprocess.run(["pandoc", str(SRC), "-f", "markdown", "-o", str(raw), f"--reference-doc={REF}",
                        f"--resource-path={HERE}"], check=True)
        out = Path(tmp) / "out.docx"
        with zipfile.ZipFile(raw) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "word/styles.xml":
                    data = restyle_styles(data.decode()).encode()
                elif item.filename == "word/document.xml":
                    data = box_header(data.decode()).encode()
                zout.writestr(item, data)
        shutil.copy(out, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
