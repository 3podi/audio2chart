"""Every argparse attribute a tool reads has to be one it declared.

fit_tier.py read `args.min_tier` in main() and never called add_argument for it, so
every invocation died with AttributeError before doing any work. Nothing caught it,
because nothing runs these tools in CI and the failure needs no unusual input -- the
script simply had never been run once since the line was written.

This is a static check over the repository's top-level tools: parse each file, collect
the names passed to add_argument (and any explicit dest=), collect every args.<name>
read, and require the second set to be inside the first.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMESPACE_NAMES = {"args", "arguments"}


def declared_and_used(tree: ast.AST) -> tuple[set[str], dict[str, int]]:
    declared: set[str] = set()
    used: dict[str, int] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args):
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                declared.add(first.value.lstrip("-").replace("-", "_"))
            for keyword in node.keywords:
                if keyword.arg == "dest" and isinstance(keyword.value, ast.Constant):
                    declared.add(keyword.value.value)
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in NAMESPACE_NAMES):
            used.setdefault(node.attr, node.lineno)
    return declared, used


class CliArgumentTest(unittest.TestCase):

    def test_no_tool_reads_an_argument_it_never_declared(self):
        failures = []
        for path in sorted(ROOT.glob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            declared, used = declared_and_used(tree)
            if not declared:
                continue            # not an argparse tool; nothing to check
            for name, line in sorted(used.items(), key=lambda item: item[1]):
                if name not in declared:
                    failures.append(f"{path.name}:{line} reads args.{name}, "
                                    f"which is never declared")
        self.assertEqual([], failures, "\n" + "\n".join(failures))

    def test_the_check_can_actually_fail(self):
        """A guard that cannot fail is not a guard."""
        tree = ast.parse(
            "p.add_argument('--kept')\n"
            "args = p.parse_args()\n"
            "print(args.kept, args.forgotten)\n")
        declared, used = declared_and_used(tree)
        self.assertEqual({"kept"}, declared)
        self.assertIn("forgotten", used)
        self.assertNotIn("forgotten", declared)


if __name__ == "__main__":
    unittest.main()
