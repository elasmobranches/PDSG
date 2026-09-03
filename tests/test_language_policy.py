"""The release's own prose is English; verbatim-ported bodies keep theirs.

README.md states that as a policy, and a policy nobody checks drifts. It
already had: the sentence was written while eight files this project wrote
still carried Korean docstrings and comments, including the CLI's own module
docstring and the whole module docstring of `tools/replay.py` -- the tool every
verification document points a reader at.

So the rule is enforced here rather than promised. Korean may appear in
exactly the modules whose class bodies were ported byte-for-byte from the
research repository, where it is part of what "verbatim" means and where
editing it would give up the property four review rounds have checked. It may
not appear anywhere else.

The test is deliberately an equality, not a subset. A file gaining Korean is
new untranslated prose in code this release wrote. A file *losing* it means a
ported body was edited, which is the thing the port fidelity claim rests on
not happening -- so that fails here too, rather than passing quietly.
"""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Written as escapes rather than as a literal range, so that this file is
# not itself a counterexample to the rule it enforces -- which it was, on
# the first run.
HANGUL = re.compile('[\uac00-\ud7a3]')

SUFFIXES = ('.py', '.yaml', '.md')

#: Files whose Korean is carried over with a verbatim-ported body, with what
#: was ported into each. Their module headers say the same thing in English.
PORTED = {
    'chamnet/datasets/dataset.py',                  # ChamNet, from chamoe.py
    'chamnet/datasets/transforms/load_depth.py',    # LoadDepthAsChannel, ShuffleDepthChannel
    'chamnet/hooks/iter_logger.py',                 # IterLoggerHook, whole file
    'chamnet/models/backbones/convnext.py',         # the four ConvNeXt-Atto classes
    'chamnet/models/backbones/early_fusion.py',     # the four 4-channel stems
    'chamnet/models/backbones/mit.py',              # the four MiT-B0 classes
    'chamnet/models/backbones/mscan.py',            # the four MSCAN classes
    'chamnet/models/backbones/resnet.py',           # the four ResNetV1c-18 classes
    'chamnet/models/fusion.py',                     # CrossModalGating's own comment
}

#: Not source: the paper's own merged configs, committed as evidence. Nothing
#: here is this project's prose and none of it is edited.
EXEMPT_PREFIXES = ('tests/fixtures/',)

#: Directories that are output rather than source. `build/` is the one that
#: matters and the one that caught this test out: `pip install /opt/chamnet`
#: builds in place and leaves `build/lib/chamnet/...` -- *copies* of the ported
#: modules -- sitting inside the source tree. A walk that does not know that
#: reports nine ported files as unexplained Korean, which is exactly what
#: happened inside the release image while the same commit was green under the
#: test harness. The policy was never violated; the enumeration was.
BUILD_DIRECTORIES = ('build', 'dist', '.git', '.venv', '__pycache__',
                     '.pytest_cache', '.hypothesis')


def _git_listing(root):
    """What git calls this tree's source, or None outside a repository.

    `--cached --others --exclude-standard` is tracked files plus files that
    are neither tracked nor ignored: everything that is, or would be, part of
    the source. That is the question the policy asks, and it drops build
    output for free because .gitignore names it.
    """
    try:
        done = subprocess.run(
            ('git', '-C', str(root), 'ls-files', '-z',
             '--cached', '--others', '--exclude-standard'),
            capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return [name for name in done.stdout.split('\0') if name]


def _walk_listing(root):
    """The same question with no git available -- a checkout without history,
    or the installed tree inside the release image. Build output is excluded
    explicitly here, since there is no .gitignore being consulted."""
    names = []
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in BUILD_DIRECTORIES or part.endswith('.egg-info')
               for part in relative.parts[:-1]):
            continue
        names.append(relative.as_posix())
    return names


def source_files(root, use_git=True):
    """`(relative paths, whether git answered)`.

    Both paths must give the same answer on the same tree; a policy test that
    means something different depending on where it runs is not a policy test.
    `test_the_two_enumerations_agree_on_a_tree_with_build_output` pins that.
    """
    names = _git_listing(root) if use_git else None
    answered_by_git = names is not None
    if names is None:
        names = _walk_listing(root)
    return sorted(name for name in names
                  if name.endswith(SUFFIXES)
                  and not name.startswith(EXEMPT_PREFIXES)), answered_by_git


def _text_files(root=ROOT):
    for relative in source_files(root)[0]:
        path = root / relative
        if path.is_file():
            yield relative, path


def test_korean_appears_only_in_verbatim_ported_modules():
    found = {relative for relative, path in _text_files()
             if HANGUL.search(path.read_text(errors='ignore'))}

    unexpected = sorted(found - PORTED)
    assert not unexpected, (
        'these files carry Korean but were written by this release, not '
        f'ported into it: {unexpected}. Translate them -- the README says the '
        "release's own prose is English.")

    vanished = sorted(PORTED - found)
    assert not vanished, (
        f'{vanished} no longer carry the Korean they were ported with. Either '
        'a ported body was edited -- which the port fidelity claim depends on '
        'not happening -- or the file is gone. If a port was deliberately '
        'replaced, remove it from PORTED in this test and say so.')


def test_nothing_a_user_sees_is_korean():
    """The narrower rule, checked separately because it is the one that bites.

    Even inside a ported module, a string that reaches a terminal is output,
    not provenance. `chamnet/hooks/iter_logger.py` is the case that motivates
    this: 33 Korean lines, every one a comment or a docstring, and every string
    it logs already English. That is what makes leaving it alone correct, and
    this pins it.
    """
    import ast

    emitting = {'print', 'print_log', 'info', 'warning', 'error', 'debug',
                'critical', 'add_argument', 'add_parser'}
    offenders = []
    for relative, path in _text_files():
        if path.suffix != '.py':
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            strings = []
            if isinstance(node, ast.Call):
                name = getattr(node.func, 'id', None) or getattr(node.func, 'attr', None)
                if name in emitting:
                    strings = list(ast.walk(node))
            elif isinstance(node, ast.Raise):
                strings = list(ast.walk(node))
            for sub in strings:
                if (isinstance(sub, ast.Constant) and isinstance(sub.value, str)
                        and HANGUL.search(sub.value)):
                    offenders.append(f'{relative}:{sub.lineno}')

    assert not offenders, (
        f'these strings reach a user and are Korean: {sorted(set(offenders))}')


# A word in Korean, built from escapes so that this file stays compliant with
# the rule it enforces.
_KOREAN_COMMENT = '# \uc548\ub155\n'


def _tree_with_build_output(root):
    """A miniature source tree in the state `pip install .` leaves behind.

    Three kinds of non-source, each of which has to be invisible to the
    enumeration: the in-place `build/lib/` copy of a module, the `.egg-info`
    directory beside it, and the committed paper fixtures. Every one of them
    is given Korean content, because a copy of a ported module is exactly what
    the real failure was.
    """
    (root / 'chamnet' / 'models').mkdir(parents=True)
    (root / 'chamnet' / 'models' / 'ported.py').write_text(_KOREAN_COMMENT)
    (root / 'chamnet' / 'clean.py').write_text('# english only\n')

    (root / 'build' / 'lib' / 'chamnet' / 'models').mkdir(parents=True)
    (root / 'build' / 'lib' / 'chamnet' / 'models' / 'ported.py').write_text(
        _KOREAN_COMMENT)
    (root / 'chamnet.egg-info').mkdir()
    (root / 'chamnet.egg-info' / 'stale.md').write_text(_KOREAN_COMMENT)
    (root / 'tests' / 'fixtures').mkdir(parents=True)
    (root / 'tests' / 'fixtures' / 'paper.py').write_text(_KOREAN_COMMENT)

    (root / '.gitignore').write_text('build/\ndist/\n*.egg-info/\n')


def test_the_two_enumerations_agree_on_a_tree_with_build_output(tmp_path):
    """The fallback has to mean what the git path means.

    This is the test that would have caught the real defect. The suite runs
    against a tree with no `.git` in both the verification harness and the
    release image, so the git path is not the one under load there -- and the
    walk it fell back to had no idea what `build/` was. A check that quietly
    answers differently depending on where it runs is the same shape as the
    stale-bytecode hole this suite already closed: it is green for a reason
    that has nothing to do with the thing it claims to check.

    So both paths are exercised here, on the same synthetic tree, and required
    to agree. Neither may see the build copy, the egg-info, or the fixtures.
    """
    _tree_with_build_output(tmp_path)
    expected = ['chamnet/clean.py', 'chamnet/models/ported.py']

    walked, answered_by_git = source_files(tmp_path, use_git=False)
    assert answered_by_git is False
    assert walked == expected, walked

    subprocess.run(('git', 'init', '-q', str(tmp_path)), check=True)
    listed, answered_by_git = source_files(tmp_path)
    assert answered_by_git is True, (
        'git is installed but did not answer for a repository it just created')
    assert listed == expected, listed
    assert listed == walked


def test_the_enumeration_sees_this_repository_as_source(tmp_path):
    """A guard against the enumeration silently returning nothing.

    Both `found - PORTED` and `PORTED - found` are empty when the file list is
    empty, so an enumeration that broke outright would leave the policy tests
    passing while checking nothing at all.
    """
    names = [relative for relative, _ in _text_files()]
    assert len(names) > 30, names
    assert 'README.md' in names
    assert 'chamnet/cli.py' in names
    for ported in PORTED:
        assert ported in names, f'{ported} is in PORTED but not enumerated'
    assert not any(name.startswith(('build/', 'dist/')) for name in names)
