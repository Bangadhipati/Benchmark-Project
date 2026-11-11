import markdown
import os

with open('PROJECT_REPORT.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Convert markdown with extensions
html_body = markdown.markdown(text, extensions=['tables', 'fenced_code', 'codehilite', 'toc'])

html_template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Scientific Highlight Generation Benchmark - Project Report</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
<script>
    document.addEventListener("DOMContentLoaded", function() {
        renderMathInElement(document.body, {
            delimiters: [
                {left: "$$", right: "$$", display: true},
                {left: "$", right: "$", display: false}
            ]
        });
    });
</script>
<style>
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.68;
        color: #1f2937;
        max-width: 920px;
        margin: 40px auto;
        padding: 0 24px;
        background-color: #ffffff;
    }
    h1 {
        color: #1e3a8a;
        font-size: 2.1rem;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 12px;
        margin-top: 24px;
    }
    h2 {
        color: #1e40af;
        font-size: 1.45rem;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 8px;
        margin-top: 36px;
    }
    h3 {
        color: #1d4ed8;
        font-size: 1.15rem;
        margin-top: 24px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 22px 0;
        font-size: 0.95rem;
    }
    th, td {
        border: 1px solid #d1d5db;
        padding: 10px 14px;
        text-align: left;
    }
    th {
        background-color: #f3f4f6;
        font-weight: 600;
        color: #111827;
    }
    tr:nth-child(even) {
        background-color: #f9fafb;
    }
    blockquote {
        border-left: 4px solid #3b82f6;
        margin: 20px 0;
        padding: 12px 20px;
        background-color: #eff6ff;
        color: #1e3a8a;
        border-radius: 0 6px 6px 0;
    }
    code {
        background-color: #f3f4f6;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 0.9em;
    }
    pre {
        background-color: #1f2937;
        color: #f9fafb;
        padding: 16px;
        border-radius: 8px;
        overflow-x: auto;
    }
    pre code {
        background-color: transparent;
        color: inherit;
        padding: 0;
    }
    img {
        max-width: 100%;
        height: auto;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin: 18px 0;
        display: block;
    }
    hr {
        border: 0;
        border-top: 1px solid #e5e7eb;
        margin: 36px 0;
    }
    @media print {
        body {
            max-width: 100%;
            margin: 0;
            padding: 15mm;
            font-size: 11pt;
        }
        h1, h2, h3 {
            page-break-after: avoid;
        }
        table, pre, img, blockquote {
            page-break-inside: avoid;
        }
    }
</style>
</head>
<body>
""" + html_body + """
</body>
</html>"""

with open('PROJECT_REPORT.html', 'w', encoding='utf-8') as f:
    f.write(html_template)

print('SUCCESS: PROJECT_REPORT.html created.')

