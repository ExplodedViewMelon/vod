from __future__ import annotations

from typing import Any, Optional

import jinja2
from vod_tools.pipes.utils.misc import iter_examples


def template_pipe(
    batch: dict[str, Any],
    idx: Optional[list[int]] = None,  # noqa: ARG
    *,
    template: str,
    input_keys: list[str],
    output_key: str = "formatted_text",
    **kwargs: Any,
) -> dict[str, Any]:
    """Applies a jinja2 template to a batch of examples."""
    jinja_template = jinja2.Template(template)
    examples = iter_examples(batch, input_keys)
    output = (jinja_template.render(eg) for eg in examples)
    return {output_key: list(output)}
