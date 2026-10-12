"""OpenAPI helpers shared by the crawler and the ingest pipeline.

ReadMe attaches the *whole* OpenAPI file of an API group to every reference
page in that group (``ssr.oasDefinition``), not just the page's own operation.
These helpers let callers store each distinct OAS document once and work out
which operation a given page actually documents.

Dependency-free on purpose: the crawler package ships without heavy deps.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterator, Optional, Tuple

HTTP_METHODS = ("get", "post", "put", "patch", "delete", "head", "options", "trace")

_DUP_SUFFIX_RE = re.compile(r"-\d+$")
_WS_RE = re.compile(r"\s+")


def iter_operations(oas: Optional[Dict]) -> Iterator[Tuple[str, str, Dict]]:
    """Yield ``(method, path, operation)`` for every operation in ``oas``.

    ``method`` is lowercased. Non-operation path-item keys (``parameters``,
    ``servers``, ``summary``...) are skipped.
    """
    if not isinstance(oas, dict):
        return
    paths = oas.get("paths") or {}
    if not isinstance(paths, dict):
        return
    for path, item in paths.items():
        if not isinstance(item, dict):
            continue
        for method, op in item.items():
            if isinstance(method, str) and method.lower() in HTTP_METHODS and isinstance(op, dict):
                yield method.lower(), path, op


def oas_fingerprint(oas: Dict) -> str:
    """Stable 16-hex-char id for an OAS document (sha256 of canonical JSON)."""
    canonical = json.dumps(oas, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _norm(text: Any) -> str:
    return _WS_RE.sub(" ", str(text or "").strip().lower())


def find_page_operation(
    oas: Optional[Dict],
    *,
    slug: str,
    title: str = "",
    doc_api: Optional[Dict] = None,
) -> Optional[Tuple[str, str]]:
    """Return ``(method, path)`` of the operation a reference page documents.

    Tries, in order, and accepts the first rule that yields exactly one match:
      1. ``operationId == slug``
      2. ``operationId == slug`` with ReadMe's ``-N`` de-dup suffix removed
      3. normalized ``summary == title``
      4. ``doc.api.method`` + ``doc.api.url`` (when ReadMe fills ``url``)
      5. the document has exactly one operation

    Returns ``None`` when the page cannot be tied to a single operation.
    """
    ops = list(iter_operations(oas))
    if not ops:
        return None

    def _unique(cands):
        return (cands[0][0], cands[0][1]) if len(cands) == 1 else None

    rules = [
        lambda: [o for o in ops if o[2].get("operationId") == slug],
        lambda: [o for o in ops if o[2].get("operationId") == _DUP_SUFFIX_RE.sub("", slug)],
        lambda: [o for o in ops if title and _norm(o[2].get("summary")) == _norm(title)],
    ]
    api = doc_api if isinstance(doc_api, dict) else {}
    api_method = str(api.get("method") or "").lower()
    api_url = api.get("url") or ""
    if api_method and api_url:
        rules.append(lambda: [o for o in ops if o[0] == api_method and o[1] == api_url])
    rules.append(lambda: ops if len(ops) == 1 else [])

    for rule in rules:
        hit = _unique(rule())
        if hit:
            return hit
    return None
