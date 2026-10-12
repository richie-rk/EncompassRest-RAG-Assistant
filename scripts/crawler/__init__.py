"""Crawler for ICE MT Developer Connect (ReadMe-hosted) docs.

Pipeline: seed the frontier (welcome sidebars, paginated changelog index,
sitemaps) -> fetch each page once (cached) -> parse its ssr-props once ->
emit a normalized record, store its OpenAPI document once in a registry, and
enqueue everything the page reveals (sidebars, links, OAS operationIds) ->
write records, manifest, OAS registry and a completeness report atomically.
"""

__version__ = "0.2.0"
