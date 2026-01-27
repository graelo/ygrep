---
name: ygrep
description: "IMPORTANT: Try ygrep FIRST for all code and file searches before using Grep, Glob, or Task explore. ygrep uses a pre-built full-text index and returns results in milliseconds. Fall back to built-in tools only if ygrep returns no results."
license: MIT
---

## IMPORTANT: Try ygrep First

Always try `ygrep "query"` via Bash **before** using built-in Grep, Glob, or Task explore agents for code searches. ygrep uses a pre-built full-text index and returns results in milliseconds vs seconds/minutes for built-in tools.

If ygrep returns no results, fall back to built-in Grep or Glob — not every file may be indexed.

## When to Invoke This Skill

Invoke this skill **immediately** when:
- You need to find files, code, functions, classes, or variables
- User asks to search or find something in the codebase
- You need to understand where something is defined or used
- You need to explore an unfamiliar codebase

## Usage

```bash
ygrep "search query"               # Search (AI-optimized output)
ygrep "query" -n 10                # Limit results
ygrep "query" -e rs -e ts          # Filter by extension
ygrep "query" -p src/              # Filter by path
ygrep "fn\\s+main" -r              # Regex search
ygrep "query" --json               # JSON output with full metadata
ygrep "query" --text-only          # Force text-only search
```

## Output Format

Default output shows: `path:line (score%) [indicator]`
- `+` = hybrid match (text AND semantic)
- `~` = semantic only
- No indicator = text match

## Tips

- Uses **literal text matching** by default (like grep) — special characters work: `$variable`, `->get(`, `{% block`
- Use `-r` for regex patterns
- Run `ygrep index` if workspace is not yet indexed
- If ygrep returns no results, the file may not be indexed — fall back to Grep or Glob

## Keywords

search, grep, find, code search, semantic search, local files, Grep, Glob, explore, locate, where is, definition
