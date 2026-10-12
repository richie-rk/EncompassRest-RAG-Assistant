"""Tests for scripts.crawler.rewrite: ReadMe Markdown -> standalone Markdown."""
from __future__ import annotations

import json
from typing import Any, Mapping, Optional

import pytest

from scripts.crawler.rewrite import (
    MAX_REUSABLE_DEPTH,
    find_component_placeholders,
    format_table_cell,
    rewrite_body,
)

BASE_URL = "https://developer.example.com"
SECTION = "developer-connect"
ROOT = f"{BASE_URL}/{SECTION}"


def rw(body: str, reusable: Optional[Mapping[str, str]] = None) -> str:
    return rewrite_body(body, base_url=BASE_URL, section=SECTION, reusable=reusable)


def block(kind: str, data: Any, *, pretty: bool = True) -> str:
    payload = json.dumps(data, indent=2 if pretty else None, ensure_ascii=False)
    return f"[block:{kind}]\n{payload}\n[/block]" if pretty else f"[block:{kind}]{payload}[/block]"


# --- basics -----------------------------------------------------------------


@pytest.mark.parametrize("body", [None, "", "  \n\n "])
def test_empty_body(body) -> None:
    assert rw(body) == ""


def test_plain_markdown_passes_through() -> None:
    body = "# Title\n\nSome *text* with a [link](https://x.test).\n\n- a\n- b"
    assert rw(body) == body


def test_blank_line_runs_collapse_and_result_is_stripped() -> None:
    assert rw("\n\n a\n\n\n\nb\n \n\t\n\nc\n\n") == "a\n\nb\n\nc"


# --- link macros -------------------------------------------------------------


@pytest.mark.parametrize(
    "macro,url",
    [
        ("doc:get-started", f"{ROOT}/docs/get-started"),
        ("ref:get-loan", f"{ROOT}/reference/get-loan"),
        ("changelog:v25-2", f"{ROOT}/changelog/v25-2"),
        ("page:api-key", f"{ROOT}/page/api-key"),
        ("doc:webhooks#retry-logic", f"{ROOT}/docs/webhooks#retry-logic"),
    ],
)
def test_link_macros(macro: str, url: str) -> None:
    assert rw(f"See [here]({macro}) now.") == f"See [here]({url}) now."


def test_base_url_trailing_slash() -> None:
    out = rewrite_body("[a](doc:x)", base_url=BASE_URL + "/", section=SECTION)
    assert out == f"[a]({ROOT}/docs/x)"


def test_macro_spellings_found_in_real_bodies() -> None:
    body = (
        "[Workflows section]\\(ref: workflows-1) and "
        "[Order Services](<ref: order-services>) and "
        '[t](doc:x "Title")'
    )
    assert rw(body) == (
        f"[Workflows section]({ROOT}/reference/workflows-1) and "
        f"[Order Services]({ROOT}/reference/order-services) and "
        f'[t]({ROOT}/docs/x "Title")'
    )


def test_reference_definitions_and_href_attributes() -> None:
    body = '[api]: page:api-key\n\n<a href="ref:get-loan#body">Get loan</a> <a href=\'doc:x\'>x</a>'
    assert rw(body) == (
        f"[api]: {ROOT}/page/api-key\n\n"
        f'<a href="{ROOT}/reference/get-loan#body">Get loan</a> <a href=\'{ROOT}/docs/x\'>x</a>'
    )


def test_inline_code_span_rules() -> None:
    # a double-backtick span may contain a single backtick and stays verbatim
    assert rw("``a ` [x](doc:x) <Foo />`` and [y](doc:y)") == (
        f"``a ` [x](doc:x) <Foo />`` and [y]({ROOT}/docs/y)")
    # an unmatched backtick is literal and protects nothing
    assert rw("a ` b [x](doc:y)") == f"a ` b [x]({ROOT}/docs/y)"
    # spans never cross a blank line
    assert rw("`open\n\n[x](doc:y) `") == f"`open\n\n[x]({ROOT}/docs/y) `"
    # an escaped backtick does not open a span
    assert rw("\\`[x](doc:y)`") == f"\\`[x]({ROOT}/docs/y)`"


def test_macros_untouched_in_code() -> None:
    body = "Inline `[a](doc:x)` and ``(ref:y)``.\n\n```md\n[a](page:api-key)\n```"
    assert rw(body) == body


def test_root_relative_links_become_absolute() -> None:
    body = (
        f"[FAQ](/{SECTION}/docs/faq) ![logo](/img/logo.png) [t](</a b> \"T\")\n\n"
        "[api]: /x/y\n\n"
        '<a href="/p/q">q</a> <img src=\'/i.png\'>'
    )
    assert rw(body) == (
        f"[FAQ]({ROOT}/docs/faq) ![logo]({BASE_URL}/img/logo.png) [t](<{BASE_URL}/a b> \"T\")\n\n"
        f"[api]: {BASE_URL}/x/y\n\n"
        f'<a href="{BASE_URL}/p/q">q</a> <img src=\'{BASE_URL}/i.png\'>'
    )


def test_root_relative_rule_leaves_other_targets_and_code_alone() -> None:
    body = (
        "[a](https://x.test/p) [b](//cdn.test/x) [c](#anchor) [d](rel/path) "
        "GET /encompass/v3/loans (/not/a/link)\n\n"
        "`[e](/in/code)`\n\n```md\n[f](/in/fence)\n```"
    )
    assert rw(body) == body


def test_root_relative_links_inside_converted_html_block() -> None:
    html = '<p>See <a href="/developer-connect/docs/faq">FAQ</a> and <a href="doc:x">x</a>.</p>'
    out = rw(block("html", {"html": html}))
    assert out == f"See [FAQ]({ROOT}/docs/faq) and [x]({ROOT}/docs/x)."
    assert rw(out) == out


# --- legacy blocks: parameters ----------------------------------------------

REAL_PARAMETERS = {
    "data": {
        "h-0": "Endpoint",
        "h-1": "Description",
        "h-2": "URL",
        "0-0": "**Authorization** ",
        "0-1": "Used by the client application to obtain an authorization code.",
        "0-2": "<https://idp.elliemae.com/authorize>",
        "1-0": "**Token Issuance** ",
        "1-1": "Used to obtain an access token.  \nAlso used for impersonation.",
        "1-2": "<https://api.elliemae.com/oauth2/v1/token>",
    },
    "cols": 3,
    "rows": 2,
    "align": ["left", "center", "right"],
}


def test_parameters_block_becomes_gfm_table() -> None:
    out = rw("Endpoints:\n\n" + block("parameters", REAL_PARAMETERS) + "\n\nAfter.")
    assert out == (
        "Endpoints:\n\n"
        "| Endpoint | Description | URL |\n"
        "| :--- | :---: | ---: |\n"
        "| **Authorization** | Used by the client application to obtain an authorization code. "
        "| <https://idp.elliemae.com/authorize> |\n"
        "| **Token Issuance** | Used to obtain an access token.<br>Also used for impersonation. "
        "| <https://api.elliemae.com/oauth2/v1/token> |\n\n"
        "After."
    )


def test_parameters_cells_escape_pipes_keep_markdown_and_rewrite_macros() -> None:
    data = {
        "data": {
            "h-0": "Field",
            "h-1": "Notes",
            "0-0": "`a|b`",
            "0-1": "already \\| escaped; click [here](page:api-key)",
            "1-0": "- one\n- two",
            "1-1": "",
        },
        "cols": 2,
        "rows": 2,
    }
    lines = rw(block("parameters", data)).splitlines()
    assert lines[1] == "| --- | --- |"
    assert lines[2] == f"| `a\\|b` | already \\| escaped; click [here]({ROOT}/page/api-key) |"
    assert lines[3] == "| - one<br>- two |  |"


def test_parameters_infers_shape_without_cols_rows() -> None:
    data = {"data": {"h-0": "A", "h-1": "B", "0-0": "x", "2-1": "z"}}
    assert rw(block("parameters", data, pretty=False)).splitlines() == [
        "| A | B |",
        "| --- | --- |",
        "| x |  |",
        "|  |  |",
        "|  | z |",
    ]


def test_parameters_without_headers_gets_empty_header_row() -> None:
    out = rw(block("parameters", {"data": {"0-0": "only"}, "cols": 1, "rows": 1}))
    assert out.splitlines() == ["|  |", "| --- |", "| only |"]


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, ""),
        (" x ", "x"),
        ("a  \nb", "a<br>b"),
        ("a\n\nb", "a<br><br>b"),
        ("a|b", "a\\|b"),
        ("a\\|b", "a\\|b"),
        (3, "3"),
    ],
)
def test_format_table_cell(value, expected) -> None:
    assert format_table_cell(value) == expected


# --- legacy blocks: other types ---------------------------------------------


def test_callout_block() -> None:
    data = {"type": "warning", "title": "Heads up", "body": "Line one\n\nSee [x](doc:y)."}
    assert rw(block("callout", data)) == (
        f"> **Warning: Heads up**\n>\n> Line one\n>\n> See [x]({ROOT}/docs/y)."
    )


def test_callout_without_title() -> None:
    data = {"type": "info", "body": "The Try It feature can’t be executed elsewhere."}
    assert rw(block("callout", data)) == "> **Info**\n>\n> The Try It feature can’t be executed elsewhere."


def test_api_header_block() -> None:
    body = "Intro\n" + block("api-header", {"title": "Try It!"}) + "\nNext paragraph"
    assert rw(body) == "Intro\n\n## Try It!\n\nNext paragraph"


def test_image_block_caption_name_and_multiple_images() -> None:
    data = {
        "images": [
            {"image": ["https://files.test/a.png", "a.png", 993], "caption": "API [Reference] page"},
            {"image": ["https://files.test/b.png", "Fallback name", ""]},
            {"image": ["https://files.test/c d.png", "", ""], "align": "left"},
            {"caption": "No URL, caption only"},
        ]
    }
    assert rw(block("image", data, pretty=False)) == (
        "![API \\[Reference\\] page](https://files.test/a.png)\n\n"
        "![Fallback name](https://files.test/b.png)\n\n"
        "![](https://files.test/c%20d.png)\n\n"
        "No URL, caption only"
    )


def test_image_block_alt_text_in_third_slot() -> None:
    # Real shapes: [url, "", alt], [url, file name, alt] (+ caption), and the
    # older [url, name, width, height, colour] where the third slot is a size.
    data = {
        "images": [
            {"image": ["https://f.test/a.png", "", "Sandbox Iframes Illustrated"]},
            {"image": ["https://f.test/b.png", "b.png", "Tasks - Preview"], "caption": "Tasks > Preview"},
            {"image": ["https://f.test/c.png", "c.png", 993, 403, "#f4f4f4"]},
            {"image": ["https://f.test/d.png", "d.png", "800px"]},
        ]
    }
    assert rw(block("image", data)) == (
        "![Sandbox Iframes Illustrated](https://f.test/a.png)\n\n"
        "![Tasks > Preview](https://f.test/b.png)\n\n"
        "![c.png](https://f.test/c.png)\n\n"
        "![d.png](https://f.test/d.png)"
    )


def test_code_block_entries_languages_and_names() -> None:
    data = {
        "codes": [
            {"code": "curl https://x.test\n\n\n  -H 'a'", "language": "curl", "name": "Request"},
            {"code": "<Loan/>\n<CompanyNetWorth>0</CompanyNetWorth>", "language": "xml"},
            {"code": "```\nnested fence\n```", "language": "markdown"},
            {"code": "", "language": "json"},
        ]
    }
    assert rw(block("code", data)) == (
        "**Request**\n\n```curl\ncurl https://x.test\n\n\n  -H 'a'\n```\n\n"
        "```xml\n<Loan/>\n<CompanyNetWorth>0</CompanyNetWorth>\n```\n\n"
        "````markdown\n```\nnested fence\n```\n````"
    )


def test_html_block_is_converted_not_deleted() -> None:
    html = (
        "<style>.x{color:red}</style><script>track()</script>"
        "<h2>Authentication</h2><p>Use <a href=\"doc:get-an-api-key\">an API key</a>"
        " and <b>OAuth</b>.</p>"
        "<table><tr><th>Grant</th><th>Use</th></tr><tr><td>password</td><td>server</td></tr></table>"
    )
    out = rw("Before\n\n" + block("html", {"html": html}) + "\n\nAfter")
    assert out == (
        "Before\n\n## Authentication\n\n"
        f"Use [an API key]({ROOT}/docs/get-an-api-key) and **OAuth**.\n\n"
        "| Grant | Use |\n| --- | --- |\n| password | server |\n\nAfter"
    )


def test_html_block_escaped_xml_survives_placeholder_stripping() -> None:
    out = rw(block("html", {"html": "<p>Send &lt;Loan/&gt; to the API.</p>"}))
    assert out == "Send \\<Loan/> to the API."


def test_embed_block() -> None:
    data = {"url": "https://youtu.be/xyz", "title": "Intro video", "html": "<p>Watch first</p>"}
    assert rw(block("embed", data)) == "[Intro video](https://youtu.be/xyz)\n\nWatch first"
    assert rw(block("embed", {"url": "https://x.test/y"})) == "[https://x.test/y](https://x.test/y)"


def test_unknown_block_keeps_human_readable_strings() -> None:
    data = {
        "title": "Tile title",
        "align": "left",
        "backgroundColor": "#018FF4",
        "sizing": "80%",
        "items": [{"text": "First item"}, {"text": "Second item"}, {"text": "First item"}],
    }
    assert rw(block("tutorial-tile", data)) == "Tile title\n\nFirst item\n\nSecond item"


def test_known_block_with_unexpected_shape_falls_back_to_strings() -> None:
    assert rw(block("parameters", {"rows": 2, "note": "kept text"})) == "kept text"
    assert rw(block("image", {"images": "https://files.test/a.png"})) == "https://files.test/a.png"


def test_malformed_json_keeps_inner_text_verbatim() -> None:
    body = "A\n\n[block:callout]\n{\"type\": \"info\", body: Use [x](doc:y)\n[/block]\n\nB"
    assert rw(body) == f"A\n\n{{\"type\": \"info\", body: Use [x]({ROOT}/docs/y)\n\nB"


def test_many_blocks_in_one_body() -> None:
    body = "\n\n".join(
        [
            block("api-header", {"title": "One"}),
            block("parameters", {"data": {"h-0": "H", "0-0": "c"}, "cols": 1, "rows": 1}),
            block("callout", {"type": "info", "body": "note"}),
            block("image", {"images": [{"image": ["https://f.test/i.png", "i"]}]}, pretty=False),
        ]
    )
    out = rw(body)
    assert "[block:" not in out and "[/block]" not in out
    assert out == (
        "## One\n\n| H |\n| --- |\n| c |\n\n> **Info**\n>\n> note\n\n![i](https://f.test/i.png)"
    )


def test_block_inside_list_item_stays_indented() -> None:
    code = json.dumps({"codes": [{"code": "a = 1\n\nb = 2", "language": "python"}]})
    body = f"1. Step one\n\n   [block:code]\n   {code}\n   [/block]\n2. Step two"
    assert rw(body) == "1. Step one\n\n   ```python\n   a = 1\n\n   b = 2\n   ```\n2. Step two"


def test_block_inside_blockquote() -> None:
    body = '> Quote\n>\n> [block:api-header]\n> {"title": "Inside"}\n> [/block]\n'
    assert rw(body) == "> Quote\n>\n> ## Inside"


def test_block_inside_fenced_code_is_left_alone() -> None:
    body = '```text\n[block:api-header]\n{"title": "Example"}\n[/block]\n```'
    assert rw(body) == body


def test_inline_block_next_to_text() -> None:
    img = block("image", {"images": [{"image": ["https://f.test/i.png", "icon"]}]}, pretty=False)
    assert rw(f"Click {img} to continue.") == "Click ![icon](https://f.test/i.png) to continue."


# --- component placeholders ---------------------------------------------------


def test_placeholders_removed_without_reusable_content() -> None:
    body = "Intro\n\n<ReleaseTagAdded252 />\n\n<BackToWebhooksOverview/>\n\nText <Foo bar=\"x\" baz={1} /> end"
    assert rw(body) == "Intro\n\nText  end"


def test_escaped_and_non_self_closing_tags_are_not_placeholders() -> None:
    body = "Use \\<Loan/> and <CompanyNetWorth>0</CompanyNetWorth> and <Base64 encoded value>"
    assert rw(body) == body


def test_placeholders_in_code_survive_byte_for_byte() -> None:
    body = (
        "Inline `<Loan/>` here.\n\n"
        "```xml\n<Loan/>\n<CompanyNetWorth>0</CompanyNetWorth>\n<ReleaseTagAdded252 />\n\n\n\n```\n\n"
        "~~~\n<Foo />\n~~~"
    )
    assert rw(body, reusable={"ReleaseTagAdded252": "SHOULD NOT APPEAR"}) == body


def test_reusable_content_is_inlined_and_rewritten() -> None:
    reusable = {
        "ReleaseTagAdded252": "**Added in 25.2.** See [notes](changelog:25-2).",
        "BackToWebhooksOverview": "Back to [Webhooks](doc:webhooks)\n\n"
        + block("callout", {"type": "info", "body": "Retry rules apply."}),
    }
    body = "# Event\n\n<ReleaseTagAdded252 />\nDetails.\n\n<BackToWebhooksOverview />"
    assert rw(body, reusable) == (
        f"# Event\n\n**Added in 25.2.** See [notes]({ROOT}/changelog/25-2).\n\nDetails.\n\n"
        f"Back to [Webhooks]({ROOT}/docs/webhooks)\n\n> **Info**\n>\n> Retry rules apply."
    )


def test_single_line_reusable_content_inline() -> None:
    assert rw("Status: <Tag /> today", {"Tag": "Preview"}) == "Status: Preview today"


def test_multi_line_reusable_content_in_list_item_is_indented() -> None:
    reusable = {"Steps": "First\n\n```\nx\n```"}
    assert rw("- Item\n\n  <Steps />\n- Next", reusable) == "- Item\n\n  First\n\n  ```\n  x\n  ```\n- Next"


def test_reusable_content_code_is_not_rewritten_by_outer_pass() -> None:
    reusable = {"Sample": "Example:\n\n```xml\n<Loan/>\n\n\n\n(doc:x)\n```"}
    assert rw("<Sample />", reusable) == "Example:\n\n```xml\n<Loan/>\n\n\n\n(doc:x)\n```"


def test_nested_reusable_content_depth_limit() -> None:
    chain = {f"L{i}": f"level {i} <L{i + 1} />" for i in range(1, 10)}
    out = rw("<L1 />", chain)
    assert out == " ".join(f"level {i}" for i in range(1, MAX_REUSABLE_DEPTH + 1))
    assert "<L" not in out


def test_reusable_cycles_terminate() -> None:
    assert rw("<A />", {"A": "a <B />", "B": "b <A />"}) == "a b"
    assert rw("<Self />", {"Self": "me <Self />"}) == "me"


def test_builtin_mdx_components_are_rendered() -> None:
    body = '<Image src="https://f.test/x.png" alt="Diagram" />\n\n<Embed url="https://y.test" title="Demo" />'
    assert rw(body) == "![Diagram](https://f.test/x.png)\n\n[Demo](https://y.test)"


def test_find_component_placeholders() -> None:
    cell = {"data": {"h-0": "H", "0-0": "<InCell />"}, "cols": 1, "rows": 1}
    code = {"codes": [{"code": "<InCodeBlock />", "language": "xml"}]}
    body = (
        "<ReleaseTagAdded252 />\n\nText <Foo a=\"1\" /> and `<InlineCode />`\n\n"
        "```\n<Fenced />\n```\n\n"
        + block("parameters", cell)
        + "\n\n"
        + block("code", code)
        + "\n\n<ReleaseTagAdded252/> <Image src=\"https://f.test/i.png\" /> \\<Escaped />"
    )
    assert find_component_placeholders(body) == ["ReleaseTagAdded252", "Foo", "InCell"]
    assert find_component_placeholders("") == []


# --- idempotency ---------------------------------------------------------------

COMPOSITE = "\n\n".join(
    [
        "# Guide\n\nSee [key](page:api-key), [W]\\(ref: workflows-1) and [O](<ref: order-services>).",
        "<Intro />",
        block("parameters", REAL_PARAMETERS),
        block("callout", {"type": "danger", "title": "Stop", "body": "Read [this](doc:x).\n\n```\na\n\n\nb\n```"}),
        block("html", {"html": "<h3>HTML part</h3><p>&lt;Loan/&gt; and <a href='ref:y'>y</a></p><pre>p\n\n\nq</pre>"}),
        block("code", {"codes": [{"code": "<Loan/>\n\n\n", "language": "xml", "name": "Sample"}]}),
        block("tutorial-tile", {"title": "Tile"}),
        "1. Item\n   [block:image]{\"images\":[{\"image\":[\"https://f.test/i.png\",\"\",\"\"]}]}[/block]\n2. Next",
        "> [block:api-header]\n> {\"title\": \"Quoted\"}\n> [/block]",
        "[block:broken]{nope[/block]",
        "Tail <Unknown /> text\n\n\n\n",
    ]
)
REUSABLE = {"Intro": "Intro with [link](doc:intro)\n\n<Nested />", "Nested": "```\n<X/>\n```"}


@pytest.mark.parametrize("reusable", [None, REUSABLE])
def test_idempotent(reusable) -> None:
    once = rw(COMPOSITE, reusable)
    assert "[block:" not in once and "(doc:" not in once and "(page:" not in once
    assert rw(once, reusable) == once


def test_idempotent_on_real_world_shapes() -> None:
    body = (
        "> ```json Date Source  = \"event\"\n>         \"dateSrc\": {\n>\n>\n>         }\n> ```\n\n"
        "   ```curl Subscribe\n    curl --location 'https://x.test'\n   ```\n"
        "Use `GET /planCodes?planCodeType=\\<opening | closing>` now.\n\n\n\n"
        "<span style=\"color:LimeGreen\">12345678</span>  \nnext line"
    )
    once = rw(body)
    assert rw(once) == once
    assert "```json Date Source  = \"event\"\n>         \"dateSrc\": {\n>\n>\n>         }\n> ```" in once


# --- regressions ------------------------------------------------------------


def test_code_fenced_inside_nested_containers_is_left_alone() -> None:
    for body in ('- > ```xml\n  > <Loan />\n  > <Field id="1"/>\n  > ```',
                 "- - ```\n    <Loan />\n    ```"):
        assert rw(body) == body


def test_legacy_block_inside_quoted_list_item_is_converted() -> None:
    payload = json.dumps({"type": "warning", "title": "Careful", "body": "Read this."}, indent=2)
    quoted = "\n".join("  > " + line for line in payload.split("\n"))
    out = rw(f"- > [block:callout]\n{quoted}\n  > [/block]")
    assert out == "- > > **Warning: Careful**\n  > >\n  > > Read this."


def test_parenthesised_prose_is_not_a_link_macro() -> None:
    body = "Borrower details are on the 1003 form (page: 2) and in RFC 7231 (ref: 6.5.1)."
    assert rw(body) == body
    # real macros keep working in every link spelling
    assert rw("[a](doc:x) [b](<ref: y>) [c]\\(ref: z) ![i](page:p) (<page:q>)") == (
        f"[a]({ROOT}/docs/x) [b]({ROOT}/reference/y) [c]({ROOT}/reference/z) "
        f"![i]({ROOT}/page/p) ({ROOT}/page/q)")


def test_multi_line_reusable_content_in_table_cell_and_heading() -> None:
    table = "| Field | Notes |\n|---|---|\n| loanId | <LoanIdNote /> |\n| other | x |"
    out = rw(table, {"LoanIdNote": "Required.\n\nMust be a GUID | UUID."})
    assert out == ("| Field | Notes |\n|---|---|\n"
                   "| loanId | Required.<br><br>Must be a GUID \\| UUID. |\n| other | x |")
    assert rw("## <Title />\n\nbody", {"Title": "Loan API\nv3"}) == "## Loan API v3\n\nbody"
    # next to prose it is still split out into a block of its own
    assert "Intro \n\na\n\nb\n\n end" == rw("Intro <Note /> end", {"Note": "a\n\nb"})


def test_image_component_keeps_its_visible_caption() -> None:
    out = rw('<Image src="/p.png" alt="screenshot" caption="Figure 3: Set the ACS URL" />')
    assert out == f"![Figure 3: Set the ACS URL]({BASE_URL}/p.png)"
    assert rw('<Image src="/p.png" alt="screenshot" />') == f"![screenshot]({BASE_URL}/p.png)"


def test_escaped_backslash_before_backtick_opens_a_code_span() -> None:
    # "\\\\" is an escaped backslash, so the backtick after it opens a span
    body = "Install to C:\\\\`<Encompass />` and see `<Loan />`"
    assert rw(body) == body
    # an odd number escapes the first backtick: the rest of the run can still open
    assert rw("a \\```<X />`` b") == "a \\```<X />`` b"
    assert rw("\\`<X />` [x](doc:y)") == f"\\`` [x]({ROOT}/docs/y)"


def test_lone_surrogate_in_legacy_block_json_is_replaced() -> None:
    out = rw('[block:callout]{"type": "info", "body": "Great job \\ud83d"}[/block]')
    assert out == "> **Info**\n>\n> Great job \ufffd"
    out.encode("utf-8")
