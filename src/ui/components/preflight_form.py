"""
Generic preflight form for analysis use cases.

Called by `gradio_app.py` to render the mandatory fields of each use
case dynamically as Gradio components.

Interface:

    components, get_inputs, validate = render_preflight_form(checker)

  - `components`: list of Gradio input components (in the order of the
    requirements). Aggregated in `analysis_components_flat` and passed
    to event handlers, which slice the values.
  - `get_inputs`: function that returns a dict {field: value} from the
    current component values (for server-side validation if needed).
  - `validate`: function `(values_tuple) -> (ok, error_msg)` that checks
    the values against the requirements.

The components are NOT wrapped in a container — the caller decides
(typically a `gr.Group` with `visible=False` as the use-case panel).
"""

from __future__ import annotations

from typing import Callable

import gradio as gr

from src.pipeline.analysis_pipeline import PreflightChecker, Requirement
from src.ui.i18n import tr


def render_preflight_form(
    checker: PreflightChecker,
) -> tuple[list, Callable, Callable]:
    """Render a checker's mandatory fields as Gradio components.

    Args:
        checker: PreflightChecker with `get_requirements()`.

    Returns:
        (components, get_inputs, validate)
          - components: list[gr.Component] in requirement order
          - get_inputs(*values) -> dict[field, value]
          - validate(*values) -> tuple[bool, str]
    """
    requirements: list[Requirement] = checker.get_requirements()
    components: list = []

    for req in requirements:
        comp = _make_component(req)
        components.append(comp)

    def get_inputs(*values) -> dict:
        """Build a {field: value} dict from the positional values."""
        return {
            req.field: val
            for req, val in zip(requirements, values)
        }

    def validate(*values) -> tuple[bool, str]:
        """Validate the values against the requirements."""
        inputs = get_inputs(*values)
        ok, errors = checker.validate(inputs)
        return ok, " · ".join(errors) if errors else ""

    return components, get_inputs, validate


def _make_component(req: Requirement):
    """Build the matching Gradio component for a requirement.

    Labels, placeholders and choice labels are English in the registry and
    translated here, in the language of the page being built.
    """
    label = tr(req.label) + (" *" if req.required else "")
    placeholder = tr(req.placeholder) if req.placeholder else ""
    info = placeholder or None

    if req.kind == "choice" and req.choices:
        return gr.Dropdown(
            choices=[(tr(c[0]), c[1]) if isinstance(c, (tuple, list)) else c
                     for c in req.choices],
            value=None,
            label=label,
            info=info,
            elem_id=f"preflight-{req.field}",
        )

    if req.kind == "textarea":
        return gr.Textbox(
            label=label,
            placeholder=placeholder,
            info=info,
            lines=4,
            max_lines=10,
            elem_id=f"preflight-{req.field}",
        )

    # Default: single-line textbox
    return gr.Textbox(
        label=label,
        placeholder=placeholder,
        info=info,
        lines=1,
        elem_id=f"preflight-{req.field}",
    )
