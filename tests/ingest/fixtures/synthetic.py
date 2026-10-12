"""Synthetic OpenAPI documents and JSONL records for the ingest tests.

The OAS document is small but exercises what the renderer has to get right:
path-level parameters (one via `$ref`, one overridden by the operation),
servers at document / path / operation level, a shared `$ref` schema large
enough to be de-duplicated, error responses, a deprecated operation, a
`type` list with a null member, and an operation without `operationId`.

`to_new_contract` converts legacy records (embedded `oas`) into the crawler's
current data contract (`oas_id` + `operation` + a separate registry).
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scripts.crawler.oas import find_page_operation, iter_operations, oas_fingerprint

BASE = "https://docs.example.com/developer-connect"


def make_oas(title: str = "Widgets API", *, prefix: str = "/widgets") -> Dict[str, Any]:
    widget_props = {
        name: {"type": "string", "description": f"The widget's {name.replace('_', ' ')} as stored by the service."}
        for name in ("id", "name", "colour", "owner_id", "created_at", "updated_at", "status", "notes")
    }
    item = f"{prefix}/{{widgetId}}"
    return {
        "openapi": "3.0.1",
        "info": {"title": title, "version": "1.0", "description": "Manage widgets and their lifecycle."},
        "servers": [
            {"url": "https://api.example.com", "description": "Production"},
            {"url": "", "description": ""},
        ],
        "paths": {
            item: {
                "servers": [{"url": "https://path.example.com"}],
                "parameters": [
                    {"$ref": "#/components/parameters/WidgetId"},
                    {"name": "X-Trace", "in": "header", "description": "path-level trace header",
                     "schema": {"type": "string"}},
                ],
                "get": {
                    "operationId": "get-widget",
                    "summary": "Get Widget",
                    "description": "Retrieves a single widget.",
                    "parameters": [
                        {"name": "X-Trace", "in": "header", "required": True,
                         "description": "operation-level trace header", "schema": {"type": "string"}},
                        {"name": "view", "in": "query", "description": "Projection",
                         "schema": {"type": ["string", None]}},
                    ],
                    "responses": {
                        "200": {
                            "description": "OK",
                            "content": {"application/json": {
                                "schema": {"$ref": "#/components/schemas/Widget"},
                                "examples": {
                                    "Model": {"value": {"id": "w-1", "name": "Sprocket"}},
                                    "Live": {"value": {"id": "w-2", "name": "Cog"}},
                                },
                            }},
                        },
                        "404": {"description": "Not Found", "content": {"application/json": {
                            "schema": {"$ref": "#/components/schemas/Error"}}}},
                    },
                },
                "patch": {
                    "operationId": "update-widget",
                    "summary": "Update Widget",
                    "deprecated": True,
                    "servers": [{"url": "https://op.example.com", "description": "Op"}],
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Widget"}}},
                    },
                    "responses": {"204": {"description": "No Content"},
                                  "400": {"description": "Bad Request"}},
                },
            },
            prefix: {
                "get": {
                    "operationId": "list-widgets",
                    "summary": "List Widgets",
                    "responses": {"200": {"description": "OK", "content": {"application/json": {
                        "schema": {"type": "array", "items": {"$ref": "#/components/schemas/Widget"}}}}}},
                },
                "post": {
                    "operationId": "create-widget",
                    "summary": "Create Widget",
                    "requestBody": {"content": {"application/json": {
                        "schema": {"$ref": "#/components/schemas/Widget"},
                        "example": {"name": "Sprocket"}}}},
                    "responses": {"201": {"description": "Created"}},
                },
            },
            f"{item}/archive": {
                "post": {"summary": "Archive Widget",
                         "responses": {"200": {"description": "OK"}}},
            },
        },
        "components": {
            "parameters": {
                "WidgetId": {"name": "widgetId", "in": "path", "required": True,
                             "description": "Widget identifier", "schema": {"type": "string"}},
            },
            "schemas": {
                "Widget": {"type": "object", "title": "Widget", "required": ["id"], "properties": widget_props},
                "Error": {"type": "object", "properties": {"errorCode": {"type": "string"},
                                                           "details": {"type": "string"}}},
            },
        },
    }


def record(
    slug: str,
    *,
    title: Optional[str] = None,
    type: str = "endpoint",
    kind: str = "reference",
    oas: Optional[Dict] = None,
    body: str = "",
    url: Optional[str] = None,
    hidden: bool = False,
    discovered_via: Optional[str] = None,
    breadcrumb: Optional[List[str]] = None,
    doc_api: Optional[Dict] = None,
) -> Dict[str, Any]:
    """A legacy-format record (embedded `oas`, as in the committed dataset)."""
    section = "docs" if kind == "guide" else "reference"
    r: Dict[str, Any] = {
        "source": "developer_connect",
        "kind": kind,
        "type": type,
        "slug": slug,
        "title": title if title is not None else slug.replace("-", " ").title(),
        "url": url or f"{BASE}/{section}/{slug}",
        "breadcrumb": list(breadcrumb or []),
        "hidden": hidden,
        "body_md": body,
    }
    if oas is not None:
        r["oas"] = oas
        r["oas_public_url"] = "@developer-connect/v1.0#abc"
    if doc_api is not None:
        r["doc_api"] = doc_api
    if discovered_via is not None:
        r["discovered_via"] = discovered_via
    return r


def to_new_contract(records: Sequence[Dict]) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Legacy records -> (records with oas_id/operation, registry {oas_id: oas})."""
    registry: Dict[str, Dict] = {}
    out: List[Dict] = []
    for r in records:
        r = copy.deepcopy(r)
        oas = r.pop("oas", None)
        if isinstance(oas, dict) and oas:
            oas_id = oas_fingerprint(oas)
            registry[oas_id] = oas
            r["oas_id"] = oas_id
            hit = find_page_operation(oas, slug=r.get("slug") or "", title=r.get("title") or "",
                                      doc_api=r.get("doc_api"))
            if hit is None:
                r["operation"] = None
            else:
                op = next(o for m, p, o in iter_operations(oas) if (m, p) == hit)
                r["operation"] = {"method": hit[0], "path": hit[1],
                                  "operationId": op.get("operationId"), "summary": op.get("summary")}
        out.append(r)
    return out, dict(sorted(registry.items()))
