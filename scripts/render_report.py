"""Render this project's Markdown report using an installed Chrome/Edge browser.

No Python report dependencies are needed. The converter intentionally supports
the report's headings, paragraphs, tables, fenced code, figures and references;
it is not a general-purpose Markdown renderer. Local document links stay plain
text in the PDF to avoid embedding workstation-specific file URLs.
"""
from __future__ import annotations

import argparse
import html
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def inline(text: str) -> str:
    pattern = r"`[^`]+`|\*\*.+?\*\*|\[[^\]]+\]\([^)]+\)|\*[^*]+\*"
    result, end = [], 0
    for match in re.finditer(pattern, text):
        result.append(html.escape(text[end:match.start()]))
        token = match.group()
        if token.startswith("`"):
            result.append("<code>" + html.escape(token[1:-1]) + "</code>")
        elif token.startswith("**"):
            result.append("<strong>" + inline(token[2:-2]) + "</strong>")
        elif token.startswith("["):
            label, target = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", token).groups()
            result.append(f'<a href="{html.escape(target, quote=True)}">{inline(label)}</a>'
                          if target.startswith("https://") else inline(label))
        else:
            result.append("<em>" + inline(token[1:-1]) + "</em>")
        end = match.end()
    result.append(html.escape(text[end:]))
    return "".join(result)


def report_html(source: Path) -> str:
    lines = source.read_text(encoding="utf-8").splitlines()
    blocks, index = [], 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("```"):
            code = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                code.append(lines[index])
                index += 1
            blocks.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
        elif line.startswith("#"):
            prefix, title = line.split(" ", 1)
            level = len(prefix)
            page_break = level == 2 and title.startswith(("3.", "4.", "6.", "7.", "8.", "Appendix:"))
            blocks.append(f'<h{level} class="{"page-break" if page_break else ""}">{inline(title)}</h{level}>')
        elif line == "---":
            pass
        elif line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    rows.append(cells)
                index += 1
            head = "<tr>" + "".join("<th>" + inline(cell) + "</th>" for cell in rows[0]) + "</tr>"
            body = "".join("<tr>" + "".join("<td>" + inline(cell) + "</td>" for cell in row) + "</tr>" for row in rows[1:])
            blocks.append("<table><thead>" + head + "</thead><tbody>" + body + "</tbody></table>")
            continue
        elif line.startswith("!["):
            alt, target = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line).groups()
            image = (source.parent / target).resolve()
            if not image.is_file():
                raise ValueError(f"Missing figure: {target}")
            caption_index = index + 1
            while caption_index < len(lines) and not lines[caption_index].strip():
                caption_index += 1
            caption = ""
            if caption_index < len(lines) and lines[caption_index].startswith("*Figure "):
                caption = "<figcaption>" + inline(lines[caption_index].strip()[1:-1]) + "</figcaption>"
                index = caption_index
            blocks.append(f'<figure><img src="{image.as_uri()}" alt="{html.escape(alt, quote=True)}">{caption}</figure>')
        elif re.match(r"\d+\. ", line):
            items = []
            while index < len(lines) and re.match(r"\d+\. ", lines[index]):
                items.append("<li>" + inline(re.sub(r"^\d+\. ", "", lines[index])) + "</li>")
                index += 1
            blocks.append("<ol>" + "".join(items) + "</ol>")
            continue
        else:
            paragraph = [line]
            while index + 1 < len(lines) and lines[index + 1].strip():
                index += 1
                paragraph.append(lines[index].strip())
            blocks.append("<p>" + inline(" ".join(paragraph)) + "</p>")
        index += 1
    style = """
    @page { size: A4; margin: 15mm 18mm 17mm;
      @bottom-right { content: counter(page); font: 8pt Arial; color: #52616b; } }
    * { box-sizing: border-box; }
    body { font: 10.5pt/1.25 Arial, sans-serif; color: #172c39; margin: 0; }
    h1 { font-size: 22pt; line-height: 1.1; margin: 0 0 9pt; }
    h2 { font-size: 13pt; line-height: 1.2; margin: 13pt 0 6pt; break-after: avoid; }
    p { margin: 0 0 7pt; orphans: 3; widows: 3; }
    a { color: #12628c; text-decoration: none; }
    table { border-collapse: collapse; width: 100%; margin: 7pt 0 9pt; font-size: 9pt; break-inside: avoid; }
    th, td { border-bottom: 0.5pt solid #d4dce1; padding: 4pt 4pt; text-align: left; }
    th { background: #eaf0f3; font-weight: bold; }
    tr { break-inside: avoid; }
    code { font: 8.8pt Consolas, monospace; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #f1f5f6; padding: 6pt; margin: 6pt 0 8pt; break-inside: avoid; }
    figure { margin: 9pt 0; break-inside: avoid; }
    figure img { width: 100%; max-height: 65mm; object-fit: contain; }
    figcaption { font-size: 9pt; line-height: 1.2; color: #405563; margin-top: 4pt; }
    li { margin-bottom: 5pt; } ol { padding-left: 18pt; }
    .page-break { break-before: page; }
    """
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Cloud Workload Forecasting &amp; Analysis</title><style>' + style + "</style></head><body>" + "\n".join(blocks) + "</body></html>"


def find_browser() -> Path:
    for name in ("google-chrome", "chromium", "chromium-browser", "msedge", "chrome"):
        found = shutil.which(name)
        if found:
            return Path(found)
    for folder in (os.environ.get("PROGRAMFILES", ""), os.environ.get("PROGRAMFILES(X86)", "")):
        for relative in ("Google/Chrome/Application/chrome.exe", "Microsoft/Edge/Application/msedge.exe"):
            candidate = Path(folder) / relative
            if candidate.is_file():
                return candidate
    raise ValueError("No local Chrome/Chromium/Edge browser found; retain the Markdown report")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("reports/final_report.md"))
    parser.add_argument("--output", type=Path, default=Path("reports/final_report.pdf"))
    parser.add_argument("--browser", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    browser = args.browser or find_browser()
    with tempfile.TemporaryDirectory(prefix=".report-render-", dir=args.source.resolve().parent.parent) as work:
        directory = Path(work)
        document = directory / "report.html"
        document.write_text(report_html(args.source.resolve()), encoding="utf-8")
        command = [str(browser), "--headless", "--disable-gpu", "--in-process-gpu", "--no-sandbox",
                   "--disable-software-rasterizer", "--no-first-run", "--disable-background-networking",
                   "--no-pdf-header-footer", f"--user-data-dir={directory / 'profile'}",
                   f"--print-to-pdf={output}", document.as_uri()]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if result.returncode or not output.is_file():
            raise RuntimeError("Browser PDF generation failed: " + result.stderr[-1500:])
    print(output)


if __name__ == "__main__":
    main()
