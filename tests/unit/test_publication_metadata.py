"""Publication-metadata integrity checks (CITATION.cff, pyproject.toml).

These do not verify the *content* is correct (that's a human judgment call
made once when the metadata was set), only that it is well-formed, internally
consistent, and that a DOI actually looks like a DOI -- catching the class of
copy-paste error this release's audit found (a stale DOI, a repository URL
using an underscore instead of the actual hyphenated GitHub name).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

try:
    import tomllib  # Python >= 3.11
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[2]
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")


def _load_citation_cff() -> dict:
    with open(REPO_ROOT / "CITATION.cff", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_pyproject() -> dict:
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def test_citation_cff_parses_and_has_required_fields():
    cff = _load_citation_cff()
    for key in ("cff-version", "title", "type", "authors", "repository-code", "license"):
        assert key in cff, f"CITATION.cff missing required key {key!r}"
    assert cff["type"] == "software"
    assert isinstance(cff["authors"], list) and len(cff["authors"]) > 0
    for author in cff["authors"]:
        assert "family-names" in author and "given-names" in author


def test_citation_cff_preferred_citation_has_valid_doi():
    cff = _load_citation_cff()
    assert "preferred-citation" in cff, "CITATION.cff missing preferred-citation"
    pc = cff["preferred-citation"]
    for key in ("title", "authors", "journal", "year", "doi"):
        assert key in pc, f"preferred-citation missing {key!r}"
    assert DOI_RE.match(str(pc["doi"])), f"malformed DOI: {pc['doi']!r}"


def test_repository_url_is_consistent_across_metadata_files():
    cff = _load_citation_cff()
    pyproject = _load_pyproject()

    cff_url = cff["repository-code"]
    pyproject_url = pyproject["project"]["urls"]["Repository"]

    assert cff_url == pyproject_url, (
        f"CITATION.cff repository-code ({cff_url!r}) does not match "
        f"pyproject.toml's Repository URL ({pyproject_url!r})"
    )
    # The actual GitHub remote is hyphenated (dl-fault-analysis), not
    # underscored -- this repo has previously had that mismatch.
    assert "dl-fault-analysis" in cff_url
    assert "dl_fault_analysis" not in cff_url


def test_pyproject_declares_a_license():
    pyproject = _load_pyproject()
    assert "license" in pyproject["project"]


def test_software_version_is_consistent_across_metadata_files():
    """The citable software version must not drift from the package version.

    CITATION.cff's `version` is what a citation manager reports; pyproject's
    is what pip reports. A release where these disagree cites something that
    was never published under that number.
    """
    cff = _load_citation_cff()
    pyproject = _load_pyproject()

    assert "version" in cff, "CITATION.cff must declare a version for a citable release"
    cff_version = str(cff["version"])
    pyproject_version = str(pyproject["project"]["version"])

    assert cff_version == pyproject_version, (
        f"CITATION.cff version ({cff_version!r}) does not match "
        f"pyproject.toml version ({pyproject_version!r})"
    )
    assert re.match(r"^\d+\.\d+\.\d+$", cff_version), (
        f"expected a semantic version, got {cff_version!r}"
    )
