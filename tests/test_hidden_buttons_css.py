"""
The trigger buttons that JavaScript clicks for chat actions (adopt, litcheck)
are invisible. They must not catch mouse clicks: with a larger button height
they once lay over the mode dropdown, so clicking the dropdown switched the
mode to the bibliography check instead of opening it.
"""

import re

from src.ui.css import CUSTOM_CSS


def _hidden_btn_rule() -> str:
    m = re.search(r"\.hidden-btn[^{]*\{([^}]*)\}", CUSTOM_CSS)
    assert m, ".hidden-btn rule missing"
    return m.group(1)


def test_hidden_buttons_ignore_the_mouse():
    assert re.search(r"pointer-events:\s*none\s*!important", _hidden_btn_rule())


def test_hidden_buttons_take_no_room():
    rule = _hidden_btn_rule()
    for prop in ("min-height: 0", "min-width: 0", "height: 1px", "width: 1px"):
        assert prop in rule, prop
