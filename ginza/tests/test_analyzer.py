import subprocess as sp
from functools import partial
from pathlib import Path

import pytest

from ginza.analyzer import Analyzer

TEST_TEXT = "#コメント\n今日はかつ丼を食べた。\n明日は東京で蕎麦を食べる。明後日は酒が飲みたい。"

run_cmd = partial(sp.run, encoding="utf-8", stdout=sp.PIPE)


@pytest.fixture(scope="module")
def input_file(tmpdir: Path) -> Path:
    file_path = (tmpdir / "test_input.txt").resolve()
    with open(file_path, "w") as fp:
        print(TEST_TEXT, file=fp)
    yield file_path
    file_path.unlink()


@pytest.fixture
def analyzer() -> Analyzer:
    default_params = dict(
        model_path=None,
        ensure_model=None,
        split_mode='A',
        hash_comment='print',
        output_format='conllu',
        require_gpu=False,
        disable_sentencizer=False
    )
    yield Analyzer(**default_params)


class TestAnalyzer:

    def test_model_path(self, mocker, analyzer):
        spacy_load_mock = mocker.patch('spacy.load')
        analyzer.model_path = 'ja_ginza'
        analyzer.set_nlp()
        spacy_load_mock.assert_called_once_with('ja_ginza')

    def test_ensure_model(self, mocker, analyzer):
        spacy_load_mock = mocker.patch('spacy.load')
        analyzer.ensure_model = 'ja_ginza_electra'
        analyzer.set_nlp()
        spacy_load_mock.assert_called_once_with('ja_ginza_electra')

    def test_splitmode(self, mocker, analyzer):
        pass

    def test_hash_comment(self, mocker, analyzer):
        pass

    def test_output_format(self, mocker, analyzer):
        spacy_load_mock = mocker.patch('spacy.load')
        analyzer.output_format = 'mecab'
        analyzer.set_nlp()
        spacy_load_mock.assert_not_called()

    def test_require_gpu(self, mocker, analyzer):
        require_gpu_mock = mocker.patch('spacy.require_gpu')
        analyzer.require_gpu = 1
        analyzer.set_nlp()
        require_gpu_mock.assert_called_once()

    def test_disable_sentencizer(self, mocker, analyzer):
        pass

