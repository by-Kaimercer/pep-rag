"""Parse PEP .rst files into section-aware chunks with metadata."""
import json
import re

from .config import CHUNK_WORDS, CHUNKS_PATH, DATA_DIR, RAW_DIR

HEADING_CHARS = set("=-~^*+#\"'`")


def parse_header(lines):
    """RFC-822 style header at the top of every PEP. Returns (meta, body_start)."""
    meta, key = {}, None
    for i, line in enumerate(lines):
        if not line.strip():
            return meta, i + 1
        if line[0].isspace() and key:
            meta[key] += " " + line.strip()
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            meta[key] = val.strip()
    return meta, len(lines)


def is_underline(line, title):
    s = line.rstrip()
    return (
        len(s) >= 3
        and len(set(s)) == 1
        and s[0] in HEADING_CHARS
        and len(s) >= len(title.rstrip()) - 1
        and title.strip() != ""
    )


def split_sections(body_lines):
    """Yield (section_path, text). Heading levels are assigned in order of first appearance, as RST does."""
    levels, path = [], []
    current, buf = "Preamble", []
    i = 0
    sections = []
    while i < len(body_lines):
        line = body_lines[i]
        nxt = body_lines[i + 1] if i + 1 < len(body_lines) else ""
        # overline-style headings (=====\nTitle\n=====) are treated like underline headings
        if is_underline(line, nxt) and i + 2 < len(body_lines) and body_lines[i + 2].rstrip() == line.rstrip():
            line, nxt = nxt, body_lines[i + 2]
            i += 1
        if line.strip() and not line[0].isspace() and is_underline(nxt, line):
            sections.append((current, "\n".join(buf)))
            ch = nxt.strip()[0]
            if ch not in levels:
                levels.append(ch)
            depth = levels.index(ch)
            path = path[:depth] + [line.strip()]
            current = " > ".join(path)
            buf = []
            i += 2
            continue
        buf.append(line)
        i += 1
    sections.append((current, "\n".join(buf)))
    return [(s, t) for s, t in sections if t.strip()]


def pack_paragraphs(text, limit=CHUNK_WORDS):
    paras = [re.sub(r"[ \t]+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    paras = [p for p in paras if p and not p.startswith(".. ")]
    chunks, cur, n = [], [], 0
    for p in paras:
        words = p.split()
        if len(words) > limit:  # very long paragraph or code block: hard split
            if cur:
                chunks.append("\n\n".join(cur))
                cur, n = [], 0
            for j in range(0, len(words), limit):
                chunks.append(" ".join(words[j : j + limit]))
            continue
        if n + len(words) > limit and cur:
            chunks.append("\n\n".join(cur))
            cur, n = [], 0
        cur.append(p)
        n += len(words)
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def parse_pep(path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    meta, start = parse_header(lines)
    try:
        number = int(meta.get("PEP", "").strip())
    except ValueError:
        return []
    title = meta.get("Title", "")
    out = []
    for section, text in split_sections(lines[start:]):
        if section.lower().startswith(("copyright", "footnotes", "references")):
            continue
        for piece in pack_paragraphs(text):
            out.append(
                {
                    "id": f"pep-{number:04d}#{len(out)}",
                    "pep": number,
                    "title": title,
                    "status": meta.get("Status", ""),
                    "type": meta.get("Type", ""),
                    "created": meta.get("Created", ""),
                    "python_version": meta.get("Python-Version", ""),
                    "section": section,
                    "text": piece,
                }
            )
    return out


def contextual_text(c):
    """Chunk text prefixed with document-level context, so a chunk like 'This was rejected because...' still says which PEP it is from."""
    pv = f", Python {c['python_version']}" if c["python_version"] else ""
    return f"PEP {c['pep']}: {c['title']} ({c['status']}, {c['type']}{pv}). Section: {c['section']}.\n{c['text']}"


def load_chunks():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    DATA_DIR.mkdir(exist_ok=True)
    chunks = []
    files = sorted(RAW_DIR.glob("pep-*.rst"))
    for p in files:
        chunks.extend(parse_pep(p))
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    peps = len({c["pep"] for c in chunks})
    words = sum(len(c["text"].split()) for c in chunks)
    print(f"{len(files)} files -> {peps} PEPs -> {len(chunks)} chunks ({words / len(chunks):.0f} words avg)")


if __name__ == "__main__":
    main()
