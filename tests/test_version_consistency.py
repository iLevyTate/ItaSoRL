"""Release metadata agrees with itself: CITATION.cff and pyproject.toml carry one version,
CITATION's date-released is a real date, and requirements.txt lists the same runtime
dependencies as pyproject.toml. Git-free, so it also runs inside the extracted supplement.

Plain text parsing on purpose: tomllib is 3.11+ and CI still runs Python 3.10, and PyYAML is
not a dependency."""
import datetime
import os
import re

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _read(name: str) -> str:
    with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
        return fh.read()


def _pyproject_version() -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', _read("pyproject.toml"), re.M)
    assert m, "pyproject.toml has no version line"
    return m.group(1)


def _cff_field(field: str) -> str:
    m = re.search(rf'^{field}:\s*"?([^"\n]+?)"?\s*$', _read("CITATION.cff"), re.M)
    assert m, f"CITATION.cff has no {field} line"
    return m.group(1)


def _name(spec: str) -> str:
    """The distribution name of a requirement, lower case, without version or extras."""
    return re.split(r"[<>=!~\[;\s]", spec.strip(), maxsplit=1)[0].lower().replace("_", "-")


def _pyproject_dependencies() -> set[str]:
    block = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", _read("pyproject.toml"), re.M | re.S)
    assert block, "pyproject.toml has no dependencies list"
    return {_name(s) for s in re.findall(r'"([^"]+)"', block.group(1))}


def _requirements() -> set[str]:
    lines = (ln.split("#", 1)[0].strip() for ln in _read("requirements.txt").splitlines())
    return {_name(ln) for ln in lines if ln and not ln.startswith("-")}


def test_citation_version_equals_pyproject():
    assert _cff_field("version") == _pyproject_version()


def test_citation_date_released_is_a_date():
    datetime.date.fromisoformat(_cff_field("date-released"))


def test_requirements_and_pyproject_list_the_same_dependencies():
    assert _requirements() == _pyproject_dependencies()
    assert "threadpoolctl" in _pyproject_dependencies()
