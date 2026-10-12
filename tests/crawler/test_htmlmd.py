"""Tests for scripts.crawler.htmlmd: HTML -> Markdown and the fence scanner."""
from __future__ import annotations

import re

import pytest

from scripts.crawler.htmlmd import (
    collapse_blank_lines,
    fence_for,
    html_to_markdown,
    split_fenced_code,
)


# --- split_fenced_code ------------------------------------------------------


def _code(text: str) -> list:
    return [chunk for is_code, chunk in split_fenced_code(text) if is_code]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "plain\ntext\n",
        "a\n```\nx\n```\nb",
        "```js\nx\n```",
        "a\n~~~\nx\n~~~\n",
        "a\n```\nunclosed\n\n\n",
        "\n\n```\n```\n\n",
    ],
)
def test_split_fenced_code_round_trips(text: str) -> None:
    assert "".join(chunk for _, chunk in split_fenced_code(text)) == text


def test_code_chunk_spans_fence_lines_only() -> None:
    parts = split_fenced_code("a\n\n```json\n{}\n```\n\nb")
    assert parts == [(False, "a\n\n"), (True, "```json\n{}\n```"), (False, "\n\nb")]


def test_closing_fence_must_match_char_and_length() -> None:
    text = "````md\n```\ninner\n```\n~~~\n````\nafter"
    assert _code(text) == ["````md\n```\ninner\n```\n~~~\n````"]


def test_tilde_fence_and_longer_closing_fence() -> None:
    assert _code("~~~~\n```\n~~~~~~\nx") == ["~~~~\n```\n~~~~~~"]


def test_backtick_info_string_with_backtick_is_not_a_fence() -> None:
    assert _code("```inline``` text\nmore") == []


def test_fences_inside_blockquotes_and_list_items() -> None:
    quoted = "> ```json\n> {\n>\n> }\n> ```\nafter"
    assert _code(quoted) == ["> ```json\n> {\n>\n> }\n> ```"]
    listed = "1. Step\n   ```bash\n   echo\n   ```\n2. Next"
    assert _code(listed) == ["   ```bash\n   echo\n   ```"]
    marker = "- ```\n  x\n  ```\n"
    assert _code(marker) == ["- ```\n  x\n  ```"]


def test_unclosed_fence_runs_to_end() -> None:
    assert split_fenced_code("a\n```\nb\n\n\nc") == [(False, "a\n"), (True, "```\nb\n\n\nc")]


# --- collapse_blank_lines / fence_for ---------------------------------------


def test_collapse_blank_lines_outside_code_only() -> None:
    text = "a\n\n\n\nb\n  \n \t\nc\n```\n1\n\n\n\n2\n```\n\n\n\nd"
    assert collapse_blank_lines(text) == "a\n\nb\n\nc\n```\n1\n\n\n\n2\n```\n\nd"


def test_collapse_keeps_single_blank_lines_and_hard_breaks() -> None:
    text = "line one  \nline two\n\npara"
    assert collapse_blank_lines(text) == text


@pytest.mark.parametrize(
    "code,expected",
    [("x", "```"), ("a ``` b", "````"), ("````", "`````"), ("`x`", "```")],
)
def test_fence_for(code: str, expected: str) -> None:
    assert fence_for(code) == expected


# --- html_to_markdown -------------------------------------------------------


@pytest.mark.parametrize("html", [None, "", "   \n\t "])
def test_empty_input(html) -> None:
    assert html_to_markdown(html) == ""


def test_headings_links_emphasis_lists() -> None:
    md = html_to_markdown(
        "<h1>Title</h1><h3>Sub</h3>"
        '<p>See <a href="https://x.test/a">the docs</a>, <b>bold</b> and <i>it</i>.</p>'
        "<ul><li>one</li><li>two<ul><li>nested</li></ul></li></ul>"
        "<ol><li>first</li><li>second</li></ol>"
    )
    assert "# Title" in md
    assert "### Sub" in md
    assert "[the docs](https://x.test/a)" in md
    assert "**bold**" in md and "*it*" in md
    assert "- one" in md and "  - nested" in md
    assert "1. first" in md and "2. second" in md


def test_drops_non_content_elements_but_keeps_visible_text() -> None:
    html = """<html><head><title>Tab title</title><style>.a{color:red}</style>
    <script>var secret = 1;</script></head><body>
    <script type="text/javascript">alert('x')</script>
    <style>p { margin: 0 }</style>
    <noscript>Enable JS</noscript>
    <svg><text>svg label</text></svg>
    <iframe src="https://video.test/embed"></iframe>
    <p>Visible paragraph</p>
    <form><label>Client ID</label><input value="typed"><button>Submit</button>
    <select><option>Choice A</option></select><textarea>draft</textarea></form>
    <!-- hidden comment -->
    <div><span>Span text</span></div>
    </body></html>"""
    md = html_to_markdown(html)
    for gone in ("Tab title", "color:red", "secret", "alert", "margin", "Enable JS",
                 "svg label", "video.test", "typed", "Submit", "Choice A", "draft",
                 "hidden comment"):
        assert gone not in md
    for kept in ("Visible paragraph", "Client ID", "Span text"):
        assert kept in md


def test_keeps_all_visible_words() -> None:
    html = (
        "<div><h2>Create package event</h2><p>The <em>package</em> event fires "
        "when <strong>documents</strong> are ready.</p><table><tr><th>Field</th>"
        "<th>Meaning</th></tr><tr><td>loanId</td><td>Loan GUID</td></tr></table>"
        "<blockquote>Quoted note</blockquote><dl><dt>Term</dt><dd>Definition</dd></dl></div>"
    )
    md = html_to_markdown(html)
    words = re.findall(r"\w+", re.sub(r"<[^>]+>", " ", html))
    assert all(w in md for w in words)


def test_table_conversion() -> None:
    md = html_to_markdown(
        "<table><tr><th>Name</th><th>Value</th></tr>"
        "<tr><td>a|b</td><td>x<br>y</td></tr>"
        "<tr><td><p>p1</p><p>p2</p></td>"
        '<td><code>c</code> <img src="https://i.test/x.png" alt="cell img"></td></tr></table>'
    )
    lines = md.splitlines()
    assert lines[0] == "| Name | Value |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == r"| a\|b | x<br>y |"
    assert "p1" in lines[3] and "p2" in lines[3]
    assert "![cell img](https://i.test/x.png)" in lines[3]
    assert len(lines) == 4  # every row stays on one line


def test_headerless_table_uses_first_row_as_header() -> None:
    md = html_to_markdown("<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>")
    assert md.splitlines() == ["| a | b |", "| --- | --- |", "| c | d |"]


def test_pre_becomes_fenced_code_with_language_and_exact_content() -> None:
    html = '<pre><code class="language-json">{\n  "a": 1,\n\n\n  "b": "&lt;Loan/&gt;"\n}</code></pre>'
    md = html_to_markdown(html)
    assert md == '```json\n{\n  "a": 1,\n\n\n  "b": "<Loan/>"\n}\n```'


def test_pre_fence_is_longer_than_backticks_in_code() -> None:
    md = html_to_markdown('<pre class="lang-md">```\nx\n```</pre>')
    assert md == "````md\n```\nx\n```\n````"


def test_pre_inside_list_and_blockquote_keeps_blank_lines() -> None:
    md = html_to_markdown("<ol><li>step<pre>a\n\n\nb</pre></li></ol><blockquote><pre>q\n\n\nr</pre></blockquote>")
    assert "   a\n\n\n   b" in md
    assert "> q\n>\n>\n> r" in md


def test_images_and_data_uris() -> None:
    md = html_to_markdown(
        '<p><img src="https://i.test/a.png" alt="Diagram [v2]" title="T"></p>'
        '<p><img src="data:image/png;base64,AAAA" alt="inline pic"></p>'
    )
    assert '![Diagram \\[v2\\]](https://i.test/a.png "T")' in md
    assert "base64" not in md and "inline pic" in md


def test_tag_like_text_is_escaped_not_lost() -> None:
    md = html_to_markdown("<p>Send the &lt;Loan/&gt; element and &lt;ServiceSetup&gt;.</p><p>1 &lt; 2</p>")
    assert r"\<Loan/>" in md and r"\<ServiceSetup>" in md
    assert "1 < 2" in md


def test_underscores_not_escaped_and_javascript_links_unwrapped() -> None:
    md = html_to_markdown('<p>Use loan_id here. <a href="javascript:void(0)">Open</a></p>')
    assert "loan_id" in md
    assert "Open" in md and "javascript" not in md


def test_collapses_blank_runs_from_br_soup() -> None:
    md = html_to_markdown("<div>top</div><br><br><br><br><br><div>bottom</div>")
    assert md == "top\n\nbottom"


# --- regressions ------------------------------------------------------------


def test_document_without_closing_head_keeps_its_body() -> None:
    # HTML5 lets "</head>" be omitted; html.parser then nests <body> in <head>.
    doc = ('<html><head><meta charset="utf-8"><title>Doc</title><style>h1{color:red}</style>'
           '<body><h1>Getting an API key</h1><p>Sign in and open Settings.</p></body></html>')
    assert html_to_markdown(doc) == "# Getting an API key\n\nSign in and open Settings."
    assert html_to_markdown("<head><style>p{}</style>\n<h1>Lender MFA</h1><p>Enable MFA.</p>") == (
        "# Lender MFA\n\nEnable MFA.")
    # a well-formed head still contributes nothing
    assert html_to_markdown("<html><head><title>T</title><base href='/'></head>"
                            "<body><p>Hi</p></body></html>") == "Hi"


@pytest.mark.parametrize("text", [
    "- > ```xml\n  > <Loan />\n  > ```",
    "- - ```\n    <Loan />\n    ```",
    "1. > - ```\n   >   <Loan />\n   >   ```",
])
def test_fence_opening_after_nested_container_markers(text: str) -> None:
    assert _code(text) == [text]


def test_pre_at_the_start_of_nested_containers_round_trips_as_code() -> None:
    for html in ("<ul><li><ul><li><pre>&lt;Loan /&gt;</pre></li></ul></li></ul>",
                 "<ul><li><blockquote><pre>&lt;Loan /&gt;</pre></blockquote></li></ul>"):
        md = html_to_markdown(html)
        assert len(_code(md)) == 1 and "<Loan />" in _code(md)[0], md
