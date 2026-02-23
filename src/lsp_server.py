from __future__ import annotations

from typing import List, Optional

from lark.exceptions import LarkError
from lsprotocol.types import (
    TEXT_DOCUMENT_DID_CHANGE,
    TEXT_DOCUMENT_DID_CLOSE,
    TEXT_DOCUMENT_DID_OPEN,
    TEXT_DOCUMENT_HOVER,
    Diagnostic,
    DiagnosticSeverity,
    DidChangeTextDocumentParams,
    DidCloseTextDocumentParams,
    DidOpenTextDocumentParams,
    Hover,
    HoverParams,
    MarkupContent,
    MarkupKind,
    Position,
    Range,
)
from pygls.lsp.server import LanguageServer

from src import __version__ as fuse_version
from src.parser import fuse_parser


class FuseLanguageServer(LanguageServer):
    pass


fuse_server = FuseLanguageServer("fuse-language-server", fuse_version)


def _find_decl_position_in_text(text: str, decl_name: str):
    """Best-effort: find the line/column for `node|model <decl_name>` in `text`.

    Returns (line, column, snippet) where line/column are 1-based, or (None,None,None).
    """
    import re

    if not decl_name:
        return None, None, None
    lines = text.splitlines()
    pattern = re.compile(rf"\b(?:node|model)\s+{re.escape(decl_name)}\b")
    for i, ln in enumerate(lines):
        m = pattern.search(ln)
        if m:
            col = m.start() + 1
            # include one-line snippet for context
            start = max(0, i - 1)
            end = min(len(lines), i + 2)
            snippet = "\n".join(lines[start:end])
            return i + 1, col, snippet
    return None, None, None


def collect_diagnostics(text: str) -> List[Diagnostic]:
    # Parse-level diagnostics first
    try:
        ast = fuse_parser.parse(text)
    except LarkError as error:
        return [_diagnostic_from_error(error)]
    except Exception as error:  # pragma: no cover - fallback
        return [_diagnostic_from_error(error)]

    # If parsing succeeded, attempt a conservative lowering pass to surface
    # lowering errors as editor diagnostics (best-effort; non-blocking).
    try:
        from src.lowering.main import FuseLowerer

        fl = FuseLowerer()
        try:
            # Only perform a lowering pass to catch structural errors; do not
            # serialize or emit large artifacts here.
            fl.lower(ast)
        except Exception as e:
            # If lowering produced a LoweringError, map it to a diagnostic
            from src.lowering.utils import LoweringError

            if isinstance(e, LoweringError):
                ln, col, snippet = _find_decl_position_in_text(text, e.function)
                # Map into LSP Position (0-based)
                if ln is None:
                    ln = 1
                if col is None:
                    col = 1
                start = Position(max(0, ln - 1), max(0, col - 1))
                end = Position(start.line, start.character + 1)
                msg = f"Lowering error: {e}"
                if snippet:
                    msg = msg + "\n\n" + snippet
                return [
                    Diagnostic(
                        range=Range(start=start, end=end),
                        message=msg,
                        severity=DiagnosticSeverity.Error,
                    )
                ]
    except Exception:
        # Best-effort: do not allow diagnostics to fail due to lowering step
        pass

    # No diagnostics
    return []


def _diagnostic_from_error(error: BaseException) -> Diagnostic:
    line = _safe_int(getattr(error, "line", None))
    column = _safe_int(getattr(error, "column", None))
    if line is None or column is None:
        line = getattr(error, "line_no", None)
        column = getattr(error, "column_no", None)
    if line is None:
        line = 1
    if column is None:
        column = 1
    start = Position(max(0, line - 1), max(0, column - 1))
    end = Position(start.line, start.character + 1)
    # Translate certain Lark errors into short, user-friendly diagnostics
    msg = str(error)
    if "Unexpected end-of-input" in msg:
        msg = "Unasserted EOF"
    return Diagnostic(
        range=Range(start=start, end=end),
        message=msg,
        severity=DiagnosticSeverity.Error,
    )


def _safe_int(value: Optional[int]) -> Optional[int]:
    if isinstance(value, int):
        return value
    return None


def _publish_diagnostics(
    server: FuseLanguageServer, uri: str, text: str
) -> None:
    diagnostics = collect_diagnostics(text)
    server.publish_diagnostics(uri, diagnostics)


@fuse_server.feature(TEXT_DOCUMENT_DID_OPEN)
def did_open(
    server: FuseLanguageServer, params: DidOpenTextDocumentParams
) -> None:
    _publish_diagnostics(
        server, params.text_document.uri, params.text_document.text
    )


@fuse_server.feature(TEXT_DOCUMENT_DID_CHANGE)
def did_change(
    server: FuseLanguageServer, params: DidChangeTextDocumentParams
) -> None:
    doc = server.workspace.get_document(params.text_document.uri)
    _publish_diagnostics(server, params.text_document.uri, doc.source)


@fuse_server.feature(TEXT_DOCUMENT_HOVER)
def hover(server: FuseLanguageServer, params: HoverParams) -> Optional[Hover]:
    doc = server.workspace.get_document(params.text_document.uri)
    lines = doc.source.splitlines()
    position = params.position
    if position.line >= len(lines):
        return None
    line = lines[position.line]
    word = _word_at(column=position.character, line=line)
    if not word:
        return None
    contents = MarkupContent(
        kind=MarkupKind.Markdown, value=f"Fuse symbol **{word}**"
    )
    hover_range = Range(
        start=Position(position.line, max(0, position.character - len(word))),
        end=Position(position.line, position.character),
    )
    return Hover(contents=contents, range=hover_range)


@fuse_server.feature(TEXT_DOCUMENT_DID_CLOSE)
def did_close(
    server: FuseLanguageServer, params: DidCloseTextDocumentParams
) -> None:
    server.publish_diagnostics(params.text_document.uri, [])


def _word_at(column: int, line: str) -> str:
    start = min(max(0, column), len(line))
    while start > 0 and (line[start - 1].isalnum() or line[start - 1] == "_"):
        start -= 1
    end = min(max(0, column), len(line))
    while end < len(line) and (line[end].isalnum() or line[end] == "_"):
        end += 1
    return line[start:end]


def main() -> None:
    fuse_server.start_io()


if __name__ == "__main__":
    main()
