"""Regression tests for lifecycle_guard null-byte script path handling.

Background
----------
``cron/lifecycle_guard.py`` guards the terminal tool and cron job creation
against gateway-lifecycle commands.  ``_read_referenced_script()`` opens
referenced script paths with ``os.open()``; a path containing an embedded
NUL byte raises ``ValueError: embedded null byte`` (CPython path
validation), which is *not* an ``OSError`` subclass.  Before the fix the
``except OSError`` clause let that ValueError escape and crash the whole
guard (terminal tool / ``create_job``), so any command or cron script
referencing a NUL-byte path blew up the gateway session.

Fix (commit 103bddfbd5): catch ``(OSError, ValueError)`` at the
``os.open`` site and treat an invalid NUL-byte path as "nothing to scan",
consistent with the existing binary/NUL-content handling (#76762).

Contract locked in here
-----------------------
For the ``os.open`` crash site fixed in 103bddfbd5, a NUL-byte script
path must NEVER surface the raw Python ValueError.  It degrades to a
clean rejection:

* ``contains_gateway_lifecycle_command_or_referenced_script`` -> ``False``
* ``check_gateway_lifecycle`` -> returns ``None`` (no raise)
* ``_read_referenced_script`` -> ``(None, False)``

Known residual gap (out of scope for this test-only task; tracked on
board t_5bcb7b0e): a NUL byte inside the USERNAME portion of a
``~user`` path token (e.g. ``bash ~x\\x00y.sh``) still crashes the
guard at ``_resolve_terminal_script_path()`` ->
``Path(candidate).expanduser()`` (``pwd.getpwnam`` raises
``ValueError: embedded null byte``) — that call site is not guarded.
It is reachable via the terminal-tool binary-read fallback recursion
(second trigger in EVIDENCE.md) and needs a production-side guard
extension before those shapes can be tested green.

These tests cover the terminal surface (shell-executable / source /
trailing-NUL / ``-c`` payload forms), the cron surface (``script=``
values, including the ``.py`` interpreter path), and valid
absolute/relative paths as sanity checks that the guard still works
normally.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron.lifecycle_guard import (  # noqa: E402
    GatewayLifecycleBlocked,
    _read_referenced_script,
    check_gateway_lifecycle,
    contains_gateway_lifecycle_command_or_referenced_script,
)

NUL_ABS_EMBEDDED = "/tmp/scri\x00pt.sh"
NUL_ABS_TRAILING = "/tmp/script.sh\x00"
NUL_SCRIPT_EMBEDDED = "bad\x00script.sh"
NUL_SCRIPT_TRAILING = "script.sh\x00"
NUL_SCRIPT_PY = "bad\x00script.py"


class TestNullByteScriptPaths:
    """NUL-byte script paths are rejected cleanly, never with the raw
    ``ValueError: embedded null byte`` from os.open()."""

    @pytest.mark.parametrize(
        "command",
        [
            pytest.param(
                f"bash {NUL_ABS_EMBEDDED}",
                id="bash-embedded-nul",
            ),
            pytest.param(
                f"source {NUL_ABS_EMBEDDED}",
                id="source-embedded-nul",
            ),
            pytest.param(
                f"bash {NUL_ABS_TRAILING}",
                id="bash-trailing-nul",
            ),
            pytest.param(
                f"bash -c 'source {NUL_ABS_EMBEDDED}'",
                id="dash-c-payload-embedded-nul",
            ),
        ],
    )
    def test_terminal_referenced_nul_script_returns_false(self, command):
        """A terminal command whose referenced script path contains a NUL
        byte returns False (treated as "nothing to scan") instead of
        raising ValueError: embedded null byte."""
        result = contains_gateway_lifecycle_command_or_referenced_script(command)
        assert result is False

    @pytest.mark.parametrize(
        "script",
        [
            pytest.param(NUL_SCRIPT_EMBEDDED, id="cron-script-embedded-nul"),
            pytest.param(NUL_SCRIPT_TRAILING, id="cron-script-trailing-nul"),
            pytest.param(NUL_SCRIPT_PY, id="cron-python-script-embedded-nul"),
        ],
    )
    def test_cron_script_nul_value_does_not_raise(self, script):
        """A cron ``script=`` value containing a NUL byte does not crash
        ``check_gateway_lifecycle`` with ValueError; it degrades to
        scanning prompt-only content (no raise, no block)."""
        assert check_gateway_lifecycle("clean prompt", script) is None

    def test_read_referenced_script_nul_path_is_nothing_to_scan(self):
        """The exact crash site (os.open inside _read_referenced_script)
        returns (None, False) for a NUL-byte path — the "nothing to scan"
        sentinel — instead of propagating ValueError."""
        assert _read_referenced_script(Path(NUL_ABS_EMBEDDED)) == (None, False)
        assert _read_referenced_script(Path(NUL_ABS_TRAILING)) == (None, False)

    def test_nul_script_paths_never_leak_raw_valueerror(self):
        """Cross-cutting contract: none of the null-byte shapes may surface
        the raw Python ValueError from os.open (the regression this suite
        exists for)."""
        surfaces = [
            f"bash {NUL_ABS_EMBEDDED}",
            f"source {NUL_ABS_EMBEDDED}",
            f"bash {NUL_ABS_TRAILING}",
        ]
        for command in surfaces:
            assert contains_gateway_lifecycle_command_or_referenced_script(command) is False
        for script in (NUL_SCRIPT_EMBEDDED, NUL_SCRIPT_TRAILING, NUL_SCRIPT_PY):
            assert check_gateway_lifecycle("clean prompt", script) is None


class TestValidPathSanity:
    """The guard still behaves normally for real, valid script paths."""

    def _write_script(self, tmp_path, name, body):
        script = tmp_path / name
        script.write_text(body)
        return script

    def test_benign_absolute_script_returns_false(self, tmp_path):
        script = self._write_script(tmp_path, "benign.sh", "echo hi\n")
        assert contains_gateway_lifecycle_command_or_referenced_script(
            f"bash {script}"
        ) is False

    def test_malicious_absolute_script_detected(self, tmp_path):
        script = self._write_script(
            tmp_path, "evil.sh", "#!/bin/bash\nhermes gateway restart\n"
        )
        assert contains_gateway_lifecycle_command_or_referenced_script(
            f"bash {script}"
        ) is True

    def test_benign_relative_script_with_cwd_returns_false(self, tmp_path):
        self._write_script(tmp_path, "benign.sh", "echo hi\n")
        assert contains_gateway_lifecycle_command_or_referenced_script(
            "bash benign.sh", cwd=str(tmp_path)
        ) is False

    def test_malicious_relative_script_with_cwd_detected(self, tmp_path):
        self._write_script(
            tmp_path, "evil.sh", "#!/bin/bash\nhermes gateway restart\n"
        )
        assert contains_gateway_lifecycle_command_or_referenced_script(
            "bash evil.sh", cwd=str(tmp_path)
        ) is True

    def test_missing_absolute_script_is_nothing_to_scan(self, tmp_path):
        assert contains_gateway_lifecycle_command_or_referenced_script(
            f"bash {tmp_path / 'nope.sh'}"
        ) is False

    def test_direct_lifecycle_command_still_detected(self):
        assert contains_gateway_lifecycle_command_or_referenced_script(
            "hermes gateway restart"
        ) is True

    def test_plain_command_not_blocked(self):
        assert contains_gateway_lifecycle_command_or_referenced_script(
            "echo hello && ls -la"
        ) is False

    def test_cron_benign_script_does_not_raise(self, tmp_path):
        script = self._write_script(tmp_path, "benign.sh", "echo hi\n")
        assert check_gateway_lifecycle("clean prompt", str(script)) is None

    def test_cron_malicious_script_blocked(self, tmp_path):
        script = self._write_script(
            tmp_path, "evil.sh", "#!/bin/bash\nhermes gateway restart\n"
        )
        with pytest.raises(GatewayLifecycleBlocked):
            check_gateway_lifecycle("clean prompt", str(script))

    def test_cron_prompt_with_lifecycle_command_blocked(self):
        with pytest.raises(GatewayLifecycleBlocked):
            check_gateway_lifecycle("run hermes gateway restart at midnight")
