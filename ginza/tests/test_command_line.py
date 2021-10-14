import json
import os
import subprocess as sp
from multiprocessing import Queue, Process, Event
from functools import partial
from pathlib import Path
from typing import Iterable, List
from unittest.mock import Mock

import pytest

import ginza.command_line as cli

TEST_TEXT = "#コメント\n今日はかつ丼を食べた。\n明日は東京で蕎麦を食べる。明後日は酒が飲みたい。"

run_cmd = partial(sp.run, encoding="utf-8", stdout=sp.PIPE)


@pytest.fixture(scope="module")
def input_file(tmpdir: Path) -> Path:
    file_path = (tmpdir / "test_input.txt").resolve()
    with open(file_path, "w") as fp:
        print(TEST_TEXT, file=fp)
    yield file_path
    file_path.unlink()


@pytest.fixture(scope="module")
def input_files(tmpdir: Path) -> Iterable[Path]:
    paths = []
    for i, text in enumerate(TEST_TEXT.split("\n")):
        file_path = (tmpdir / f"test_input_{i}.txt").resolve()
        with open(file_path, "w") as fp:
            print(text, file=fp)
        paths.append(file_path)
    yield paths
    for file_path in paths:
        file_path.unlink()


@pytest.fixture(scope="module")
def long_input_file(tmpdir: Path) -> Iterable[Path]:
    file_path = (tmpdir / "test_long_input.txt").resolve()
    with open(file_path, "w") as fp:
        for _ in range(10):
            print(TEST_TEXT, file=fp)
    yield file_path
    file_path.unlink()


@pytest.fixture
def output_file(tmpdir: Path) -> Path:
    file_path = (tmpdir / "test_output.txt").resolve()
    file_path.touch()
    yield file_path
    file_path.unlink()


def _parse_conllu(result: str):
    # TODO: implement
    pass


def _parse_cabocha(result: str):
    # TODO: implement
    pass


def _parse_mecab(result: str):
    # TODO: implement
    pass


class TestCLIGinza:
    def test_help(self):
        for opt in ["-h", "--help"]:
            p = run_cmd(["ginza", opt])
            assert p.returncode == 0

    def test_input(self, input_file):
        # input file
        p = run_cmd(["ginza", input_file])

        # input from stdin
        p_stdin = sp.Popen(["ginza"], stdin=sp.PIPE, stdout=sp.PIPE)
        o, e = p_stdin.communicate(input=TEST_TEXT.encode())
        assert e is None
        assert o.decode("utf-8") == p.stdout

    def test_multiple_input(self, input_files, input_file):
        p_multi = run_cmd(["ginza", *input_files])
        assert p_multi.returncode == 0

        p_single = run_cmd(["ginza", input_file])
        assert p_multi.stdout == p_single.stdout

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

    @pytest.mark.parametrize(
        "split_mode, input_text, expected",
        [
            ("A", "機能性食品", ["機能", "性", "食品"]),
            ("B", "機能性食品", ["機能性", "食品"]),
            ("C", "機能性食品", ["機能性食品"]),
        ],
    )
    def test_split_mode(self, split_mode, input_text, expected):
        p = run_cmd(["ginza", "-s", split_mode], input=input_text)
        assert p.returncode == 0

        def _sub_words(lines: Iterable) -> List[str]:
            return [l.split("\t")[1] for l in lines if len(l.split("\t")) > 1]

        assert _sub_words(p.stdout.split("\n")) == expected

    @pytest.mark.parametrize(
        "hash_comment, n_sentence, n_analyzed_sentence, exit_ok",
        [
            ("print", 4, 3, True),
            ("skip", 3, 3, True),
            ("analyze", 4, 4, True),
        ],
    )
    def test_hash_comment(self, hash_comment, n_sentence, n_analyzed_sentence, exit_ok, input_file):
        def _n_sentence(lines: Iterable) -> int:
            return len(list(filter(lambda x: x.startswith("#"), lines)))

        def _n_analyzed_sentence(lines: Iterable) -> int:
            return len(list(filter(lambda x: x.startswith("# text = "), lines)))

        p = run_cmd(["ginza", "-c", hash_comment, input_file])
        assert (p.returncode == 0) is exit_ok
        assert _n_sentence(p.stdout.split("\n")) == n_sentence
        assert _n_analyzed_sentence(p.stdout.split("\n")) == n_analyzed_sentence

    def test_output_path(self, input_file, output_file):
        p_s = run_cmd(["ginza", input_file])
        p_o = run_cmd(["ginza", "-o", output_file, input_file])
        assert p_o.returncode == 0

        def _file_output():
            with open(output_file, "r") as fp:
                return [l.strip() for l in fp if l.strip()]

        def _pipe_output():
            return [l.strip() for l in p_s.stdout.split("\n") if l.strip()]

        assert _file_output() == _pipe_output()

    @pytest.mark.parametrize(
        "output_format, result_parser",
        [
            ("conllu", _parse_conllu),
            ("cabocha", _parse_cabocha),
            ("mecab", _parse_mecab),
            ("json", json.loads),
        ],
    )
    def test_output_format(self, output_format, result_parser, input_file):
        p = run_cmd(["ginza", "-c", "skip", "-f", output_format, input_file])
        assert p.returncode == 0
        try:
            result_parser(p.stdout.strip())
        except:
            pytest.fail("invalid output format.")

    def test_require_gpu(self, input_file):
        p = run_cmd(["ginza", "-g", input_file])
        gpu_available = int(os.environ.get("CUDA_VISIBLE_DEVICES", -1)) > 0
        assert (p.returncode == 0) is gpu_available

    def test_use_normalized_form(self, input_file):
        p = run_cmd(["ginza", "-n", input_file])
        lemmas = [l.split("\t")[2] for l in p.stdout.split("\n") if len(l.split("\t")) > 1]
        # 'カツ丼' is normlized_form of 'かつ丼'
        assert p.returncode == 0
        assert "カツ丼" in lemmas

    def test_disable_sentencizer(self, input_file):
        p = run_cmd(["ginza", "-d", input_file])

        def _n_analyzed_sentence(lines: Iterable) -> int:
            return len(list(filter(lambda x: x.startswith("# text = "), lines)))

        assert p.returncode == 0
        assert _n_analyzed_sentence(p.stdout.split("\n")) == 2

    def test_parallel(self, input_file):
        p = run_cmd(["ginza", "-p", "2", input_file])
        assert p.returncode == 0


class TestCLIGinzame:
    def test_ginzame(self, input_file):
        p_ginzame = run_cmd(["ginzame", input_file])
        p_ginza = run_cmd(["ginza", "-m", "ja_ginza", "-f", "2", input_file])

        assert p_ginzame.returncode == 0
        assert p_ginzame.stdout == p_ginza.stdout


class TestRun:
    def test_run_as_single_when_file_is_small(self, mocker, output_file, long_input_file):
        mocker.patch.object(cli, "MINI_BATCH_SIZE", 50)
        process = mocker.patch.object(cli, "Process")
        cli.run(parallel=2, output_path=output_file, files=[long_input_file])
        process.assert_not_called()

    def test_run_as_single_when_input_is_a_tty(self, mocker, output_file, long_input_file):
        i = 0

        def f_mock_input():
            nonlocal i
            if i >= 1:
                raise KeyboardInterrupt
            else:
                i += 1
                return "今日はいい天気だ"

        mocker.patch.object(cli, "MINI_BATCH_SIZE", 5)
        mocker.patch("ginza.command_line.sys.stdin.isatty", return_value=True)
        input_mock = mocker.patch.object(cli, "input", side_effect=f_mock_input)
        process_mock = mocker.patch.object(cli, "Process")
        cli.run(parallel=2, output_path=output_file, files=None)
        assert input_mock.call_count == 2
        process_mock.assert_not_called()

    def test_parallel(self, mocker, output_file, long_input_file):
        mocker.patch.object(cli, "MINI_BATCH_SIZE", 5)
        process_obj_mock = mocker.Mock(spec=Process)
        process_obj_mock.start = mocker.Mock()
        process_obj_mock.join = mocker.Mock()
        process_mock = mocker.patch.object(cli, "Process", return_value=process_obj_mock)
        queue_mock = mocker.patch.object(cli, "Queue")
        event_mock = mocker.patch.object(cli, "Event")
        cli.run(parallel=2, output_path=output_file, files=[long_input_file])
        assert queue_mock.call_count == 2
        assert process_mock.call_count == 1 + 2 + 1
        assert event_mock.call_count == 1 + 2

    @pytest.mark.parametrize(
        "output_format",
        ["conllu", "cabocha", "mecab", "json"],
    )
    def test_parallel_output_same_as_single(self, output_format, mocker, tmpdir, long_input_file):
        mocker.patch.object(cli, "MINI_BATCH_SIZE", 5)

        out_single = tmpdir / "single_output.txt"
        if out_single.exists():
            out_single.unlink()
        cli.run(
            parallel=1,
            output_path=out_single,
            output_format=output_format,
            files=[long_input_file],
            ensure_model="ja_ginza",
        )

        out_parallel = tmpdir / "parallel_output.txt"
        if out_parallel.exists():
            out_parallel.unlink()
        try:
            cli.run(
                parallel=2,
                output_path=out_parallel,
                output_format=output_format,
                files=[long_input_file],
                ensure_model="ja_ginza",
            )
        except:
            pytest.fail("parallel run failed")

        def f_len(path):
            return int(run_cmd(["wc", "-l", path]).stdout.split()[0])

        assert f_len(out_single) == f_len(out_parallel)
