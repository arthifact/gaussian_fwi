"""The walkthrough document must quote this repository, not an older one.

Every excerpt it shows is recorded in docs/build_walkthrough.py together with
the file it came from. These checks fail when the source moves on without the
document following, which is how the document drifted out of date before.
"""

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
if DOCS.is_dir():
    sys.path.insert(0, str(DOCS))


def dedent(lines):
    pad = min((len(x) - len(x.lstrip()) for x in lines if x.strip()), default=0)
    return [x[pad:].rstrip() if x.strip() else "" for x in lines]


@unittest.skipUnless((DOCS / "build_walkthrough.py").is_file(),
                     "document sources are absent from an installed wheel")
class WalkthroughExcerptTests(unittest.TestCase):
    def excerpts(self):
        import build_walkthrough

        return build_walkthrough

    def test_every_block_appears_verbatim_and_contiguously_in_its_source(self):
        module = self.excerpts()
        for label, (relative, expected) in module.EXCERPTS.items():
            with self.subTest(excerpt=label):
                source = (ROOT / relative).read_text().splitlines()
                want = [line.rstrip() for line in expected]
                windows = (dedent(source[i:i + len(want)])
                           for i in range(len(source) - len(want) + 1))
                self.assertTrue(
                    any(window == want for window in windows),
                    f"{label!r} is no longer a contiguous block in {relative}",
                )

    def test_individually_quoted_lines_still_exist(self):
        module = self.excerpts()
        for label, (relative, line) in module.SINGLE_LINES.items():
            with self.subTest(line=label):
                stripped = [x.strip() for x in (ROOT / relative).read_text().splitlines()]
                self.assertIn(line, stripped, f"{label!r} is no longer in {relative}")

    def test_wrapped_display_form_matches_the_source_line(self):
        module = self.excerpts()
        source = [x.strip() for x in (ROOT / module.WRAPPED_SOURCE).read_text().splitlines()]
        for exact, shown in module.WRAPPED.items():
            with self.subTest(line=exact[:40]):
                self.assertIn(exact, source, "the wrapped line left its source file")
                self.assertEqual(ast.dump(ast.parse(exact)),
                                 ast.dump(ast.parse("\n".join(shown))),
                                 "the wrapped form no longer means the same thing")

    def test_figures_the_document_embeds_are_present(self):
        for name in ("gaussian-geometry", "gaussian-field", "population-edits", "macro-detail"):
            with self.subTest(figure=name):
                self.assertTrue((DOCS / f"figures/{name}.png").is_file(),
                                f"{name}.png is missing from docs/figures")


if __name__ == "__main__":
    unittest.main()
