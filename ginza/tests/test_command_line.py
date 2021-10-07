import sys
import subprocess as sp
import tempfile
from functools import partial
from pathlib import Path

import spacy
import pytest

import ginza.command_line as cli

TEST_TEXT = "すもももももももものうち"


run_cmd = partial(sp.run, encoding="utf-8", stdout=sp.PIPE)


@pytest.fixture(scope="module", autouse=True)
def download_model():
    run_cmd([sys.executable, "-m", "pip", "install", "ja-ginza"])
    run_cmd([sys.executable, "-m", "pip", "install", "ja-ginza-electra"])
    yield


@pytest.fixture(scope="module")
def tmpdir() -> Path:
    with tempfile.TemporaryDirectory() as dir_name:
        yield Path(dir_name)


@pytest.fixture(scope="module")
def input_file(tmpdir: Path):
    file_path = (tmpdir / "test_input.txt").resolve()
    with open(file_path, "w") as fp:
        print(TEST_TEXT, file=fp)
    yield file_path
    file_path.unlink()


@pytest.fixture
def analyzer(request) -> cli.Analyzer:
    yield cli.Analyzer(*request.params)


class TestAnalyzer:
    def test_init(analyzer):
        pass


class TestCLI:
    def test_help(self):
        for opt in ["-h", "--help"]:
            p = run_cmd(["ginza", opt])
            assert p.returncode == 0

    def test_input(self, input_file):
        p_stdin = sp.Popen(["ginza"], stdin=sp.PIPE, stdout=sp.PIPE)
        o, e = p_stdin.communicate(input=TEST_TEXT.encode())
        p = run_cmd(["ginza", input_file])
        assert e is None
        assert o.decode("utf-8") == p.stdout

    # TODO: add user defined model to fixture and test it here
    @pytest.mark.parametrize(
        "model_path, exit_ok",
        [
            ("ja_ginza", True),
            ("not-exist-model", False),
        ],
    )
    def test_model_path(self, model_path, exit_ok, input_file):
        p = run_cmd(["ginza", "-b", model_path, input_file])
        assert (p.returncode == 0) is exit_ok

    @pytest.mark.parametrize(
        "ensure_model, exit_ok",
        [
            ("ja_ginza", True),
            ("ja-ginza", True),
            ("ja-ginza-electra", True),
            ("ja_ginza_electra", True),
            ("ja-ginza_electra", False),
            ("not-exist-model", False),
        ],
    )
    def test_ensure_model(self, ensure_model, exit_ok, input_file):
        p = run_cmd(["ginza", "-m", ensure_model, input_file])
        assert (p.returncode == 0) is exit_ok

    def test_double_model_spcification(self, input_file):
        p = run_cmd(["ginza", "-b", "ja_ginza", "-m", "ja_ginza", input_file])
        assert p.returncode != 0

    def test_split_mode(self, input_file):
        pass

    def test_hash_comment(self, input_file):
        pass

    def test_output_path(self, input_file):
        pass

    def test_output_format(self, input_file):
        pass

    def test_use_normalized_form(self, input_file):
        pass

    def test_diable_sentencizer(self, input_file):
        pass

    def test_parallel(self, input_file):
        pass

    # def test_require_gpu(self, mocker, input_file):
    #     require_gpu_mock = mocker.patch('spacy.require_gpu')
    #     try:
    #         cli.run(require_gpu=True, files=[input_file])
    #     except:
    #         pass
    #     require_gpu_mock.assert_called_once()
