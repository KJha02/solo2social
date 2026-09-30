"""vLLM startup with an accounted, non-repairing Harmony parse-failure guard.

Only malformed Harmony generations change behavior: the normal serving path
returns their real token usage with an explicit error and no executable tools.
"""
from __future__ import annotations

import hashlib
import json
import logging
from types import SimpleNamespace

PARSE_ERROR_PREFIX = '[rung4_harmony_parse_error] '


def install_harmony_guard() -> None:
    from openai_harmony import HarmonyError
    from vllm.entrypoints.openai.parser import harmony_utils
    from vllm.tool_parsers import openai_tool_parser

    native = harmony_utils.parse_output_into_messages
    if getattr(native, '_rung4_guard', False):
        return

    def guarded(token_ids):
        # Serving supplies a sequence; materialize once for deterministic diagnostics.
        tokens = list(token_ids)
        try:
            return native(tokens)
        except HarmonyError as error:
            detail = str(error)[:240]
            # The bounded parser error describes the bad header; never decode reasoning.
            logging.getLogger(__name__).warning('%s', json.dumps(dict(
                event='rung4_harmony_parse_error', error=detail,
                token_count=len(tokens), token_sha256=hashlib.sha256(
                    json.dumps(tokens).encode()).hexdigest())))
            return SimpleNamespace(messages=[], current_channel='final',
                current_content=PARSE_ERROR_PREFIX + detail, current_recipient=None)

    guarded._rung4_guard = True
    harmony_utils.parse_output_into_messages = guarded
    # The tool parser imports this helper by value, so update that alias too.
    openai_tool_parser.parse_output_into_messages = guarded


if __name__ == '__main__':
    install_harmony_guard()
    from vllm.entrypoints.cli.main import main
    main()
