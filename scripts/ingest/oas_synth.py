"""Render OpenAPI operations (and optional ReadMe `doc.api` samples) into markdown.

Reference pages on Developer Connect carry their docs as structured OpenAPI,
with `body_md` empty for most endpoints. The retrieval pipeline can't embed
structured JSON, so we render it to LLM-friendly markdown here.

ReadMe attaches the *whole* OpenAPI file of an API group to every reference
page, so rendering "the page's OAS" naively means rendering every operation of
the group on every page (the old behaviour: ~58K chars per page, 89% duplicate
chunks). Everything here therefore works on ONE operation, or on a compact
overview of the document:

Public entry points:
    synthesize_operation_markdown(oas, method, path, *, doc_api=None) -> str
        Full reference for a single operation (servers, merged parameters,
        request body, responses, code samples). Typically 1-10K chars; large
        request/response schemas are capped (see the tunables below).
    synthesize_api_overview(oas) -> str
        Info + base URLs + one line per operation. No schemas or examples.
    synthesize_endpoint_markdown(record, oas=None) -> str
        Backward-compatible record-level helper: the page's own operation when
        it can be identified, otherwise the overview.
    synthesize_code_samples(doc_api) -> str
        Only the page's `doc.api` code samples.
    page_operation(record, oas=None) -> (method, path) | None
        Which operation a record documents: the crawler's `operation` field
        (new data contract) or `find_page_operation` on the OAS (legacy).

The renderer:
  - dereferences `$ref`s against the document (depth-capped, cycle-guarded);
  - inlines schema JSON when small, falls back to a field table when large;
  - truncates oversized example payloads;
  - appends rendered code samples from `doc.api` when ReadMe has them.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from scripts.crawler.oas import find_page_operation, iter_operations

# Tunables. They bound the size of a single operation's markdown: each schema
# is either inlined (<= MAX_INLINE_SCHEMA_CHARS) or summarized as a capped
# field table, and only one example per content block is kept.
MAX_DEREF_DEPTH = 4
MAX_INLINE_SCHEMA_CHARS = 4000      # inline as JSON if rendered size <= this
MAX_SCHEMA_TABLE_CHARS = 3500       # hard cap on field-table output per schema
MAX_EXAMPLE_CHARS = 4000
MAX_EXAMPLES_PER_BLOCK = 1  # avoid Model+Live duplication on this site
# Corpus-level sharing (see `_render_once`): schema/example blocks at least
# this large are rendered in full once and referenced afterwards.
MIN_SHARED_BLOCK_CHARS = 400
MAX_SHARED_FIELD_NAMES = 40


# --- $ref resolution ----------------------------------------------------------

def _resolve_ref(ref: str, oas: Dict) -> Optional[Dict]:
    """Resolve a JSON-pointer-ish `#/components/<bucket>/<name>` against the OAS root."""
    if not ref.startswith("#/"):
        return None
    parts = ref[2:].split("/")
    cursor: Any = oas
    for p in parts:
        # OpenAPI uses `~1` for `/` and `~0` for `~` in JSON pointers.
        p = p.replace("~1", "/").replace("~0", "~")
        if not isinstance(cursor, dict) or p not in cursor:
            return None
        cursor = cursor[p]
    return cursor if isinstance(cursor, dict) else None


def _deref(node: Any, oas: Dict, *, depth: int = 0, seen: Optional[set] = None) -> Any:
    """Recursively replace `$ref` nodes with their resolved target.

    Returns a new structure; does not mutate the input. Cycles and over-deep
    references degrade to the unresolved `$ref` dict so the caller can still
    show *something*.
    """
    if depth >= MAX_DEREF_DEPTH:
        return node
    if seen is None:
        seen = set()

    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            if ref in seen:
                return {"$ref": ref, "_cycle": True}
            target = _resolve_ref(ref, oas)
            if target is None:
                return node
            return _deref(target, oas, depth=depth + 1, seen=seen | {ref})
        return {k: _deref(v, oas, depth=depth, seen=seen) for k, v in node.items()}
    if isinstance(node, list):
        return [_deref(v, oas, depth=depth, seen=seen) for v in node]
    return node


# --- Example formatting -------------------------------------------------------

def _truncate_example(value: Any) -> str:
    text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    if len(text) <= MAX_EXAMPLE_CHARS:
        return text
    elided = len(text) - MAX_EXAMPLE_CHARS
    return text[:MAX_EXAMPLE_CHARS] + f"\n... (truncated, {elided:,} more chars)"


def _render_examples(examples: Optional[Dict], heading_level: int) -> List[str]:
    """OpenAPI `examples` is `{name: {value, summary?, description?}}`.

    Caps at `MAX_EXAMPLES_PER_BLOCK` per content block. ReadMe authors
    typically include redundant `Model Data Sample` + `Live Data Sample`
    pairs that differ only in mock data — keeping just one preserves
    retrieval value without doubling output size on every response.
    """
    if not isinstance(examples, dict) or not examples:
        return []
    h = "#" * heading_level
    out: List[str] = []
    rendered = 0
    extra = 0
    for name, ex in examples.items():
        if not isinstance(ex, dict):
            continue
        if rendered >= MAX_EXAMPLES_PER_BLOCK:
            extra += 1
            continue
        summary = ex.get("summary")
        title = name if not summary else f"{name} — {summary}"
        out.append(f"{h} {title}")
        if ex.get("description"):
            out.append(ex["description"])
        if "value" in ex:
            out.append("```json")
            out.append(_truncate_example(ex["value"]))
            out.append("```")
        rendered += 1
    if extra:
        out.append(f"_({extra} additional example(s) omitted)_")
    return out


# --- Schema rendering ---------------------------------------------------------

def _render_schema(schema: Optional[Dict]) -> List[str]:
    """Render a (already dereferenced) schema as either inline JSON or a field table."""
    if not schema:
        return []
    inline = json.dumps(schema, indent=2, ensure_ascii=False, default=str)
    if len(inline) <= MAX_INLINE_SCHEMA_CHARS:
        return ["```json", inline, "```"]
    return _render_schema_table(schema)


def _iter_properties(schema: Dict) -> Iterable[Tuple[str, Dict, bool]]:
    """Yield (name, prop_schema, required) for object-shaped schemas."""
    if not isinstance(schema, dict):
        return
    props = schema.get("properties") or {}
    if not isinstance(props, dict):
        return
    required = set(r for r in (schema.get("required") or []) if isinstance(r, str))
    for name, prop in props.items():
        if isinstance(prop, dict):
            yield name, prop, name in required


def _type_label(schema: Any) -> str:
    if not isinstance(schema, dict):
        return ""
    t = schema.get("type")
    if t == "array":
        items = schema.get("items") or {}
        inner = _type_label(items) or "object"
        return f"array<{inner}>"
    if isinstance(t, list):
        # OAS 3.1 allows `type: ["string", "null"]`; YAML/JSON quirks can also
        # leave non-string members (a bare null), so stringify defensively.
        return "|".join("null" if x is None else str(x) for x in t)
    fmt = schema.get("format")
    if t and fmt:
        return f"{t} ({fmt})"
    if t:
        return str(t)
    if "oneOf" in schema or "anyOf" in schema:
        return "oneOf"
    if "allOf" in schema:
        return "allOf"
    return ""


def _flatten_description(desc: Any) -> str:
    """Markdown table cells can't contain raw newlines or pipes. Squash both."""
    if not desc:
        return ""
    return str(desc).replace("|", "\\|").replace("\n", " ").strip()


def _render_schema_table(schema: Dict) -> List[str]:
    """Field-table representation for schemas too large to inline.

    Lists top-level fields only (nested objects show as their type label).
    Hard-caps total output at `MAX_SCHEMA_TABLE_CHARS` to keep large schemas
    (e.g., `applicationnew` with hundreds of nested objects) from overwhelming
    the synthesized markdown. Past the cap, truncates with a note.
    """
    out: List[str] = []
    obj_type = schema.get("type") if isinstance(schema, dict) else None

    if obj_type == "array" and isinstance(schema.get("items"), dict):
        out.append("_Array of:_")
        out.extend(_render_schema_table(schema["items"]))
        return out

    rows = list(_iter_properties(schema))
    if not rows:
        label = _type_label(schema) or "schema"
        if isinstance(schema, dict) and schema.get("description"):
            return [f"_{label}_ — {_flatten_description(schema['description'])}"]
        return [f"_{label}_"]

    out.append("| Field | Type | Required | Description |")
    out.append("| --- | --- | --- | --- |")

    running = sum(len(s) + 1 for s in out)
    truncated = 0
    emitted = 0
    for name, prop, required in rows:
        type_str = _type_label(prop)
        desc = _flatten_description(prop.get("description"))
        row = f"| `{name}` | {type_str} | {'yes' if required else 'no'} | {desc} |"
        if running + len(row) + 1 > MAX_SCHEMA_TABLE_CHARS:
            truncated = len(rows) - emitted
            break
        out.append(row)
        running += len(row) + 1
        emitted += 1

    if truncated:
        out.append(f"_({truncated} more field(s) omitted; see full schema in OAS)_")
    return out


# --- Parameters ---------------------------------------------------------------

def _merge_parameters(path_level: Any, op_level: Any, oas: Dict) -> List[Dict]:
    """Path-item parameters merged with operation parameters (dereferenced).

    Per OpenAPI, an operation-level parameter overrides a path-level one with
    the same (name, in). Path-level parameters come first (they are usually
    the `{id}` segments of the path), then the operation's own.
    """
    def _derefed(params: Any) -> List[Dict]:
        if not isinstance(params, list):
            return []
        out = []
        for p in params:
            d = _deref(p, oas) if isinstance(p, dict) else None
            if isinstance(d, dict):
                out.append(d)
        return out

    op_params = _derefed(op_level)
    overridden = {(p.get("name"), p.get("in")) for p in op_params}
    merged = [p for p in _derefed(path_level) if (p.get("name"), p.get("in")) not in overridden]
    return merged + op_params


def _render_parameters(parameters: List[Dict]) -> List[str]:
    """Table of already-dereferenced parameter objects."""
    if not parameters:
        return []
    out = ["### Parameters", "", "| Name | In | Required | Type | Description |",
           "| --- | --- | --- | --- | --- |"]
    for p in parameters:
        name = p.get("name") or ""
        loc = p.get("in") or ""
        required = p.get("required", False)
        schema = p.get("schema") or {}
        type_str = _type_label(schema)
        desc = _flatten_description(p.get("description"))
        out.append(f"| `{name}` | {loc} | {'yes' if required else 'no'} | {type_str} | {desc} |")
    return out


# --- Servers ------------------------------------------------------------------

def _server_list(servers: Any) -> List[str]:
    """`servers` entries as display strings; blank URLs (a ReadMe artifact) dropped."""
    if not isinstance(servers, list):
        return []
    out: List[str] = []
    for s in servers:
        if not isinstance(s, dict):
            continue
        url = str(s.get("url") or "").strip()
        if not url:
            continue
        text = f"`{url}`"
        desc = str(s.get("description") or "").strip()
        if desc:
            text += f" ({desc})"
        variables = s.get("variables")
        if isinstance(variables, dict) and variables:
            defaults = ", ".join(
                f"{k}={v.get('default')}" for k, v in variables.items()
                if isinstance(v, dict) and v.get("default") is not None
            )
            if defaults:
                text += f" [variables: {defaults}]"
        if text not in out:
            out.append(text)
    return out


def _render_servers(*candidates: Any) -> List[str]:
    """First non-empty `servers` list wins (operation > path item > document)."""
    for servers in candidates:
        shown = _server_list(servers)
        if shown:
            label = "Base URL" if len(shown) == 1 else "Base URLs"
            return [f"{label}: " + ", ".join(shown)]
    return []


# --- Request / response bodies ------------------------------------------------

def _schema_field_names(schema: Any, limit: int = MAX_SHARED_FIELD_NAMES) -> str:
    """Compact "Fields: `a`, `b` (+N more)" line for a schema shown elsewhere."""
    if isinstance(schema, dict) and schema.get("type") == "array" and isinstance(schema.get("items"), dict):
        schema = schema["items"]
    names = [name for name, _, _ in _iter_properties(schema)]
    if not names:
        return ""
    shown = ", ".join(f"`{n}`" for n in names[:limit])
    more = f" (+{len(names) - limit} more)" if len(names) > limit else ""
    return f"Fields: {shown}{more}."


def _render_once(lines: List[str], seen: Optional[Dict[str, str]], label: str,
                 summary: str = "") -> List[str]:
    """Render a large schema/example block in full only the first time it occurs.

    Operations often share a body schema (the POST request of a resource is
    the GET response of the same resource). With a `seen` ledger
    ({sha256 of block: label of where it was rendered}) later occurrences
    become a one-line pointer plus `summary`, instead of byte-identical chunks
    that crowd out other results at retrieval time.
    """
    if seen is None or not lines:
        return lines
    text = "\n".join(lines)
    if len(text) < MIN_SHARED_BLOCK_CHARS:
        return lines
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    first = seen.get(key)
    if first is None:
        seen[key] = label
        return lines
    pointer = f"_Same as the {first} (rendered there in full)._"
    return [pointer, summary] if summary else [pointer]


def _render_content_block(
    content: Dict,
    oas: Dict,
    *,
    schema_heading_level: int = 4,
    example_heading_level: int = 5,
    seen: Optional[Dict[str, str]] = None,
    where: str = "",
) -> List[str]:
    """Render the `content` map of a requestBody or response (keyed by content-type).

    `where` names the block for `_render_once` pointers, e.g.
    "200 response of `GET /path`".
    """
    if not isinstance(content, dict) or not content:
        return []
    out: List[str] = []
    sh = "#" * schema_heading_level
    for ct, body in content.items():
        if not isinstance(body, dict):
            continue
        out.append(f"Content-Type: `{ct}`")
        out.append("")
        schema = _deref(body.get("schema") or {}, oas)
        rendered = _render_schema(schema)
        if rendered:
            out.append(f"{sh} Schema")
            out.extend(_render_once(rendered, seen, f"schema of the {where}",
                                    _schema_field_names(schema)))
            out.append("")
        examples = body.get("examples") or {}
        if examples:
            out.append(f"{sh} Examples")
            out.extend(_render_once(_render_examples(_deref(examples, oas), example_heading_level),
                                    seen, f"example of the {where}"))
            out.append("")
        elif "example" in body:
            out.append(f"{sh} Example")
            out.extend(_render_once(["```json", _truncate_example(body["example"]), "```"],
                                    seen, f"example of the {where}"))
            out.append("")
    return out


# --- doc.api code samples -----------------------------------------------------

def _is_meaningful_code(code: str) -> bool:
    """Skip ReadMe stub responses like `{}`, `[]`, `null` — empty placeholders
    that the API explorer leaves behind on pages without real interactions."""
    s = (code or "").strip()
    if len(s) < 5:
        return False
    return s not in ("{}", "[]", "null", '""', "''")


def _render_doc_api_codes(doc_api: Optional[Dict]) -> List[str]:
    """Pull rendered cURL / Python / JS examples out of `doc.api` when present.

    Shape varies per page: `examples.codes`, `results.codes`, sometimes both.
    Each entry is `{language, code, name?, status?}`.
    """
    if not isinstance(doc_api, dict):
        return []
    code_blocks: List[Tuple[str, str, str]] = []  # (heading, lang, code)

    examples = (doc_api.get("examples") or {}).get("codes") or []
    for c in examples:
        if not isinstance(c, dict):
            continue
        lang = (c.get("language") or "").lower() or "text"
        code = c.get("code") or ""
        if not _is_meaningful_code(code):
            continue
        name = c.get("name") or lang
        code_blocks.append((f"{name}", lang, code))

    results = (doc_api.get("results") or {}).get("codes") or []
    for c in results:
        if not isinstance(c, dict):
            continue
        lang = (c.get("language") or "").lower() or "text"
        code = c.get("code") or ""
        if not _is_meaningful_code(code):
            continue
        status = c.get("status")
        name_bits = []
        if status is not None:
            name_bits.append(f"Response {status}")
        name_bits.append(c.get("name") or lang)
        code_blocks.append((" — ".join(name_bits), lang, code))

    if not code_blocks:
        return []

    out = ["### Code samples (rendered by ReadMe)", ""]
    for heading, lang, code in code_blocks:
        out.append(f"#### {heading}")
        out.append(f"```{lang}")
        out.append(code if len(code) <= MAX_EXAMPLE_CHARS
                   else code[:MAX_EXAMPLE_CHARS] + "\n... (truncated)")
        out.append("```")
        out.append("")
    return out


# --- Operation rendering ------------------------------------------------------

def _lookup_operation(oas: Optional[Dict], method: str, path: str) -> Optional[Tuple[Dict, Dict]]:
    """Return (path_item, operation) for `method path`, or None if absent."""
    if not isinstance(oas, dict):
        return None
    item = (oas.get("paths") or {}).get(path)
    if not isinstance(item, dict):
        return None
    m = method.lower()
    for key, op in item.items():
        if isinstance(key, str) and key.lower() == m and isinstance(op, dict):
            return item, op
    return None


def _render_operation(method: str, path: str, item: Dict, op: Dict, oas: Dict,
                      seen: Optional[Dict[str, str]] = None) -> List[str]:
    where = f"`{method.upper()} {path}`"
    out = [f"## {method.upper()} {path}", ""]
    summary = op.get("summary")
    description = op.get("description")
    if summary:
        out.append(f"**{summary}**")
        out.append("")
    if description:
        out.append(str(description).strip())
        out.append("")
    if op.get("deprecated"):
        out.append("> ⚠ This operation is marked deprecated.")
        out.append("")

    servers = _render_servers(op.get("servers"), item.get("servers"), oas.get("servers"))
    if servers:
        out.extend(servers)
        out.append("")

    parameters = _merge_parameters(item.get("parameters"), op.get("parameters"), oas)
    if parameters:
        out.extend(_render_parameters(parameters))
        out.append("")

    request_body = op.get("requestBody")
    if isinstance(request_body, dict):
        rb = _deref(request_body, oas)
        if isinstance(rb, dict) and rb.get("content"):
            out.append("### Request body")
            out.append("")
            if rb.get("required"):
                out.append("_Required._")
                out.append("")
            if rb.get("description"):
                out.append(str(rb["description"]).strip())
                out.append("")
            out.extend(_render_content_block(rb["content"], oas, seen=seen,
                                             where=f"request body of {where}"))

    responses = op.get("responses") or {}
    if isinstance(responses, dict) and responses:
        out.append("### Responses")
        out.append("")
        for code, resp in responses.items():
            if not isinstance(resp, dict):
                continue
            resp_d = _deref(resp, oas)
            if not isinstance(resp_d, dict):
                continue
            desc = str(resp_d.get("description") or "").strip()
            # 4xx/5xx responses on this site reuse a generic Error schema across
            # every operation — rendering it dozens of times bloats synthesis
            # without retrieval value. Render schemas/examples for 2xx only and
            # a single line for every other status.
            is_success = str(code).startswith("2")
            if not is_success:
                line = f"- `{code}`"
                if desc:
                    line += f" — {_flatten_description(desc)}"
                out.append(line)
                continue
            if out[-1].startswith("- `"):
                out.append("")
            head = f"#### {code}"
            if desc:
                head = f"{head} — {desc}"
            out.append(head)
            out.append("")
            if resp_d.get("content"):
                out.extend(_render_content_block(
                    resp_d["content"], oas,
                    schema_heading_level=5, example_heading_level=6,
                    seen=seen, where=f"{code} response of {where}",
                ))
        if out[-1].startswith("- `"):
            out.append("")

    return out


def _finish(parts: List[str]) -> str:
    return "\n".join(parts).rstrip() + "\n" if parts else ""


# --- Public API ---------------------------------------------------------------

def synthesize_operation_markdown(
    oas: Optional[Dict],
    method: str,
    path: str,
    *,
    doc_api: Optional[Dict] = None,
    seen_blocks: Optional[Dict[str, str]] = None,
) -> str:
    """Render ONE operation of `oas` (plus the page's `doc.api` code samples).

    `seen_blocks` is an optional corpus-wide ledger: large schema/example
    blocks already rendered by another operation are replaced by a pointer
    to it (and the ledger is updated with the blocks rendered here).
    Returns an empty string when `method path` is not in the document.
    """
    found = _lookup_operation(oas, method, path)
    if found is None:
        return ""
    item, op = found
    parts = _render_operation(method, path, item, op, oas, seen_blocks)  # type: ignore[arg-type]
    codes = _render_doc_api_codes(doc_api)
    if codes:
        parts.append("")
        parts.extend(codes)
    return _finish(parts)


def synthesize_api_overview(oas: Optional[Dict]) -> str:
    """Compact overview of an OAS document: what the API group is and what it offers.

    Used for reference overview pages, which carry the group's document but
    no operation of their own. One line per operation; no schemas or examples
    (each operation is rendered in full exactly once elsewhere).
    """
    if not isinstance(oas, dict):
        return ""
    ops = list(iter_operations(oas))
    info = oas.get("info") if isinstance(oas.get("info"), dict) else {}
    title = str(info.get("title") or "").strip()
    version = str(info.get("version") or "").strip()
    description = str(info.get("description") or "").strip()
    if not ops and not title and not description:
        return ""

    heading = title or "API"
    if version:
        heading += f" (version {version})"
    parts = [f"## {heading} — API overview", ""]
    if description:
        parts += [description, ""]
    servers = _render_servers(oas.get("servers"))
    if servers:
        parts += servers + [""]
    if ops:
        parts += [f"### Operations ({len(ops)})", ""]
        for method, path, op in ops:
            line = f"- {method.upper()} {path}"
            summary = str(op.get("summary") or "").strip()
            if summary:
                line += f" - {summary}"
            if op.get("operationId"):
                line += f" ({op['operationId']})"
            if op.get("deprecated"):
                line += " [deprecated]"
            parts.append(line)
    return _finish(parts)


def synthesize_code_samples(doc_api: Optional[Dict]) -> str:
    """Just the `doc.api` code samples of a page (cURL/JSON request + responses)."""
    return _finish(_render_doc_api_codes(doc_api))


def page_operation(record: Mapping[str, Any], oas: Optional[Dict] = None) -> Optional[Tuple[str, str]]:
    """`(method, path)` of the operation a record documents, or None.

    New-contract records carry the crawler's answer in `operation`; it is
    trusted when `oas` is unknown (e.g. during dedupe, before the registry is
    consulted) or when it names an operation present in `oas`. Otherwise fall
    back to `find_page_operation` — the same matcher the crawler uses — so
    legacy records (embedded `oas`, no `operation`) resolve identically.
    """
    if oas is None:
        oas = record.get("oas")
    op = record.get("operation")
    if isinstance(op, dict):
        method = str(op.get("method") or "").lower()
        path = op.get("path")
        if method and isinstance(path, str) and path:
            if not isinstance(oas, dict) or _lookup_operation(oas, method, path) is not None:
                return method, path
    if not isinstance(oas, dict):
        return None
    return find_page_operation(
        oas,
        slug=str(record.get("slug") or ""),
        title=str(record.get("title") or ""),
        doc_api=record.get("doc_api"),
    )


def synthesize_endpoint_markdown(record: Mapping[str, Any], oas: Optional[Dict] = None) -> str:
    """Render a reference record's own operation (or the API overview) as markdown.

    `oas` defaults to the record's embedded `oas` (legacy records); pass the
    registry document for new-contract records (`oas_id`). Never renders every
    operation in full: if the page's operation cannot be identified, the
    compact overview is returned instead. Code samples from `doc.api` are
    appended in both cases. Returns "" when nothing is renderable.
    """
    if oas is None:
        oas = record.get("oas")
    doc_api = record.get("doc_api")
    hit = page_operation(record, oas)
    if hit is not None:
        md = synthesize_operation_markdown(oas, hit[0], hit[1], doc_api=doc_api)
        if md:
            return md
    parts: List[str] = []
    overview = synthesize_api_overview(oas)
    if overview:
        parts.append(overview.rstrip())
        parts.append("")
    parts.extend(_render_doc_api_codes(doc_api))
    return _finish(parts)
