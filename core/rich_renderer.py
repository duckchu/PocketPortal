import re
import html
import wcwidth

def str_width(s: str) -> int:
    w = wcwidth.wcswidth(s)
    return w if w >= 0 else len(s)

def pad_str(s: str, target_width: int, align="left") -> str:
    current_w = str_width(s)
    pad = max(0, target_width - current_w)
    if align == "right":
        return " " * pad + s
    elif align == "center":
        left_pad = pad // 2
        right_pad = pad - left_pad
        return " " * left_pad + s + " " * right_pad
    else:
        return s + " " * pad

def format_markdown_table_html(table_text: str) -> str:
    lines = [l.strip() for l in table_text.strip().split("\n") if l.strip()]
    if not lines:
        return table_text

    rows = []
    alignments = []

    for line in lines:
        raw_cells = [c.strip() for c in line.strip("|").split("|")]
        is_sep = all(c.replace(":", "").replace("-", "") == "" for c in raw_cells)
        if is_sep:
            for c in raw_cells:
                if c.startswith(":") and c.endswith(":"):
                    alignments.append("center")
                elif c.endswith(":"):
                    alignments.append("right")
                else:
                    alignments.append("left")
            continue
        rows.append(raw_cells)

    if not rows:
        return table_text

    col_count = max(len(r) for r in rows)
    while len(alignments) < col_count:
        alignments.append("left")

    html_out = ['\n<table border="1">']
    for row_idx, r in enumerate(rows):
        html_out.append("  <tr>")
        tag = "th" if row_idx == 0 else "td"
        for i in range(col_count):
            raw_val = r[i] if i < len(r) else ""
            # Handle linebreaks inside table cells: convert <br>, <br/>, <br /> to newline
            raw_val = re.sub(r"<br\s*/?>", "\n", raw_val, flags=re.IGNORECASE)
            # Escape HTML first
            c_text = html.escape(raw_val)
            # Format bold, code, strikethrough inside cells
            c_text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", c_text)
            c_text = re.sub(r"`([^`]+)`", r"<code>\1</code>", c_text)
            c_text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", c_text)
            # Support <br> inside native table cells for Telegram sendRichMessage
            c_text = c_text.replace("\n", "<br>")
            align = alignments[i]
            html_out.append(f'    <{tag} align="{align}">{c_text}</{tag}>')
        html_out.append("  </tr>")
    html_out.append("</table>\n")
    return "\n".join(html_out)

def format_markdown_table(table_text: str) -> str:
    lines = [l.strip() for l in table_text.strip().split("\n") if l.strip()]
    if not lines:
        return table_text

    rows = []
    alignments = []

    for line in lines:
        raw_cells = [c.strip() for c in line.strip("|").split("|")]
        is_sep = all(re.match(r"^:?-+:?$", c) for c in raw_cells)
        if is_sep:
            for c in raw_cells:
                if c.startswith(":") and c.endswith(":"):
                    alignments.append("center")
                elif c.endswith(":"):
                    alignments.append("right")
                else:
                    alignments.append("left")
            continue
        rows.append(raw_cells)

    if not rows:
        return table_text

    col_count = max(len(r) for r in rows)
    while len(alignments) < col_count:
        alignments.append("left")

    col_widths = [0] * col_count
    for r in rows:
        for i in range(col_count):
            c_text = r[i] if i < len(r) else ""
            # Clean <br> for plain ascii tables
            c_text = re.sub(r"<br\s*/?>", " ", c_text, flags=re.IGNORECASE)
            col_widths[i] = max(col_widths[i], str_width(c_text))

    output_lines = []
    for row_idx, r in enumerate(rows):
        formatted_cells = []
        for i in range(col_count):
            c_text = r[i] if i < len(r) else ""
            c_text = re.sub(r"<br\s*/?>", " ", c_text, flags=re.IGNORECASE)
            align = alignments[i]
            formatted_cells.append(pad_str(c_text, col_widths[i], align))
        output_lines.append(" | ".join(formatted_cells))

        if row_idx == 0 and len(rows) > 1:
            borders = ["-" * col_widths[i] for i in range(col_count)]
            output_lines.append("-+-".join(borders))

    return "\n<pre>\n" + html.escape("\n".join(output_lines)) + "\n</pre>\n"


def _format_roleplay_paragraphs(text: str) -> str:
    """Separate interleaved *narration* and \"dialogue\" into distinct paragraphs for readability."""
    # Split between closing *narration* and opening "dialogue" / 「dialogue」
    text = re.sub(r"(\*[^*\n]+\*)[ \t]+([\"\u201c\u300c])", r"\1\n\n\2", text)
    # Split between closing "dialogue" / 」 and opening *narration*
    text = re.sub(r"([\"\u201d\u300d])[ \t]+(\*[^*\n]+\*)", r"\1\n\n\2", text)
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def convert_markdown_to_telegram_html(text: str, native_tables: bool = True) -> str:
    """
    Complies with RICH_TEXT.md for Telegram Rich Message output:
    1. Single-tier headers: All #, ##, ###, etc. -> <b>Title</b>
    2. Tables -> Native <table> (when native_tables=True) or <pre> monospace aligned table
    3. Code blocks -> <pre><code>...</code></pre>
    4. Inline code -> <code>...</code>
    5. Blockquotes -> <blockquote>...</blockquote>
    6. Bold -> <b>...</b>
    7. Italic -> <i>...</i>
    8. Strikethrough -> <s>...</s>
    9. Links -> <a href="...">...</a>
    10. Expandable blocks / details -> <details><summary>...</summary>...</details>
    """
    if not text:
        return ""

    # Step 1: Protect code blocks
    code_blocks = []
    def save_code_block(match):
        code_blocks.append(f"<pre><code>{html.escape(match.group(2).strip())}</code></pre>")
        return f"___CODEBLOCK_{len(code_blocks)-1}___\n"

    text = re.sub(r"```([a-zA-Z0-9_-]*)\n?([\s\S]*?)```", save_code_block, text)

    # Step 2: Protect inline code `...`
    inline_codes = []
    def save_inline_code(match):
        inline_codes.append(f"<code>{html.escape(match.group(1))}</code>")
        return f"___INLINECODE_{len(inline_codes)-1}___"

    text = re.sub(r"`([^`\n]+)`", save_inline_code, text)

    # Step 2.5: Extract & convert <details>...</details> and protect <blockquote expandable> blocks
    detail_blocks = []
    def save_details(match):
        block = match.group(0)
        if block.lower().startswith("<details"):
            #
            sum_m = re.search(r"<summary>(.*?)</summary>", block, re.IGNORECASE | re.DOTALL)
            summary_title = sum_m.group(1).strip() if sum_m else "Details"
            body = re.sub(r"<summary>.*?</summary>", "", block, flags=re.IGNORECASE | re.DOTALL)
            body = re.sub(r"</?details>", "", body, flags=re.IGNORECASE).strip()
            block = f"\n<blockquote expandable><b>{summary_title}</b>\n{body}</blockquote>\n"
        detail_blocks.append(block)
        return f"___DETAILS_{len(detail_blocks)-1}___\n"

    text = re.sub(r"<details>[\s\S]*?</details>", save_details, text, flags=re.IGNORECASE)
    text = re.sub(r"<blockquote\s+expandable>[\s\S]*?</blockquote>", save_details, text, flags=re.IGNORECASE)

    # Step 3: Extract and format tables
    tables = []
    def save_table(match):
        raw_text = match.group(0).strip()
        # Verify it has at least one separator row like | --- | --- |
        if re.search(r"^[ \t]*\|(?:\s*:?-+:?\s*\|)+[ \t]*$", raw_text, re.MULTILINE):
            if native_tables:
                formatted = format_markdown_table_html(raw_text)
            else:
                formatted = format_markdown_table(raw_text)
            tables.append(formatted)
            return f"___TABLE_{len(tables)-1}___\n"
        return raw_text

    # Match consecutive lines starting and ending with | but not || (spoilers)
    table_pattern = re.compile(r"((?:^[ \t]*\|(?!\|)[^\n]+\|[ \t]*(?:\n|$))+)", re.MULTILINE)
    text = table_pattern.sub(save_table, text)

    #
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)

    text = _format_roleplay_paragraphs(text)

    # Step 4: HTML escape remaining text
    text = html.escape(text)

    # Step 5: Convert headers to <b>Single Tier Header</b>
    text = re.sub(r"^[ \t]*#{1,6}[ \t]+(.+?)[ \t]*$", r"<b>\1</b>", text, flags=re.MULTILINE)

    # Step 6: Horizontal rules --- or *** -> empty line / clean separation
    text = re.sub(r"^[ \t]*[-*_]{3,}[ \t]*$", "", text, flags=re.MULTILINE)

    # Step 7: Blockquotes (> ...) -> <blockquote>...</blockquote>
    quote_pattern = re.compile(r"((?:^[ \t]*&gt;.*(?:\n|$))+)", re.MULTILINE)
    def replace_quote(match):
        q_lines = match.group(0).splitlines()
        clean = "\n".join(re.sub(r"^[ \t]*&gt;[ \t]?", "", l) for l in q_lines).strip()
        return f"<blockquote>{clean}</blockquote>\n"
    text = quote_pattern.sub(replace_quote, text)

    # Step 8: Spoilers ||text|| -> <tg-spoiler>text</tg-spoiler>
    text = re.sub(r"\|\|(.+?)\|\|", r"<tg-spoiler>\1</tg-spoiler>", text)

    # Step 9: Bold & Strikethrough
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)

    # Step 10: Italic: *text* (Allow adjacent CJK characters, avoid ASCII word/multiplication like 2*3*4)
    text = re.sub(
        r"(?<![A-Za-z0-9\*])\*([^\s\*](?:[^\*\n]*?[^\s\*])?)\*(?![A-Za-z0-9\*])",
        r"<i>\1</i>",
        text,
    )

    # Step 11: Links [text](url) -> <a href="url">text</a>
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\s\)]+)\)", r'<a href="\2">\1</a>', text)

    # Step 11.5: Convert standalone &lt;br&gt; outside tables to newlines
    text = re.sub(r"&lt;br\s*/?&gt;", "\n", text, flags=re.IGNORECASE)

    # Clean redundant blank lines (max 2 consecutive)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Step 12: Restore details, tables, inline code, and code blocks
    for idx, dt in enumerate(detail_blocks):
        text = text.replace(f"___DETAILS_{idx}___", dt)

    for idx, tbl in enumerate(tables):
        text = text.replace(f"___TABLE_{idx}___", tbl)

    for idx, inc in enumerate(inline_codes):
        text = text.replace(f"___INLINECODE_{idx}___", inc)

    for idx, cb in enumerate(code_blocks):
        text = text.replace(f"___CODEBLOCK_{idx}___", cb)

    #
    #
    if native_tables:
        raw_parts = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
        wrapped_parts = []
        for part in raw_parts:
            #
            if part.startswith("<table") or part.startswith("<pre") or part.startswith("<blockquote") or part.startswith("<details"):
                wrapped_parts.append(part)
            else:
                #
                part_with_br = part.replace("\n", "<br>")
                wrapped_parts.append(f"<p>{part_with_br}</p>")
        text = "\n".join(wrapped_parts)

    return text

def split_telegram_message(content: str, max_len=3500) -> list[str]:
    """
    Splits message into chunks under max_len respecting paragraph boundaries (\n\n or \n).
    """
    if len(content) <= max_len:
        return [content]

    sep = "\n\n"
    line_sep = "\n"

    chunks = []
    paragraphs = content.split(sep)
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) + len(sep) > max_len:
            if current_chunk:
                chunks.append(sep.join(current_chunk))
                current_chunk = []
                current_len = 0
            if len(para) > max_len:
                lines = para.split(line_sep)
                for line in lines:
                    if current_len + len(line) + len(line_sep) > max_len:
                        if current_chunk:
                            chunks.append(line_sep.join(current_chunk))
                            current_chunk = []
                            current_len = 0
                        if len(line) > max_len:
                            for i in range(0, len(line), max_len):
                                chunks.append(line[i:i+max_len])
                            continue
                    current_chunk.append(line)
                    current_len += len(line) + len(line_sep)
                continue
        current_chunk.append(para)
        current_len += len(para) + len(sep)

    if current_chunk:
        chunks.append(sep.join(current_chunk))

    return chunks
