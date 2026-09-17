import pytest
from agent_campus_assistant.tools import safe_calculate


def test_basic_arithmetic():
    assert safe_calculate("18 * 30 + 12") == 552.0
    assert safe_calculate("(10 + 5) / 3") == 5.0


def test_rejects_code_execution():
    with pytest.raises((ValueError, SyntaxError)):
        safe_calculate("__import__('os').system('echo unsafe')")
