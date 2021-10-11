# coding: utf8
import io
from multiprocessing import Pool, cpu_count
from pathlib import Path
import sys
from typing import Optional, List, Iterator

import plac
from . import force_using_normalized_form_as_lemma
from .analyzer import Analyzer

MINI_BATCH_SIZE = 100


from dataclasses import dataclass


@dataclass
class OutputContext:
    output: io.TextIOWrapper

    def __enter__(self):
        self.open()
        return self

    def __exit__(self):
        self.close()
        return self

    def open():
        raise NotImplementedError

    def close():
        raise NotImplementedError

    def write(self, message):
        print(message, file=self.output)


class StreamOutput(OutputContext):
    output = sys.stdout

    def open():
        pass

    def close():
        pass


class FileOutput(OutputContext):
    output = None
    output_path: str

    def open(self):
        self.output = open(self.output_path, "w")

    def close(self):
        self.output.close()


class JsonOutput(OutputContext):
    output = None
    output_path: str

    def open(self):
        self.output = open(self.output_path, "w")
        self.write("[")

    def close(self):
        self.write("]")
        self.output.close()


def run(
    model_path: Optional[str] = None,
    ensure_model: Optional[str] = None,
    split_mode: Optional[str] = None,
    hash_comment: str = "print",
    output_path: Optional[str] = None,
    output_format: str = "0",
    require_gpu: bool = False,
    disable_sentencizer: bool = False,
    use_normalized_form: bool = False,
    parallel: int = 1,
    files: List[str] = None,
):
    if require_gpu:
        print("GPU enabled", file=sys.stderr)
    if use_normalized_form:
        print("overriding Token.lemma_ by normalized_form of SudachiPy", file=sys.stderr)
        force_using_normalized_form_as_lemma(True)
    assert model_path is None or ensure_model is None

    if parallel <= 0:
        parallel = max(1, cpu_count() + parallel)

    pool = None

    if not files:
        if sys.stdin.isatty():
            parallel = 1
        else:
            files = [0]

    analyzer = Analyzer(
        model_path,
        ensure_model,
        split_mode,
        hash_comment,
        output_format,
        require_gpu,
        disable_sentencizer,
    )

    if output_path and output_format in ["3", "json"]:
        output_context = JsonOutput(output_path=output_path)
    elif output_path:
        output_context = FileOutput(output_path=output_path)
    else:
        output_context = StreamOutput(output_path=output_path)

    if not files:
        analyzer.set_nlp()
        try:
            with output_context as oc:
                while True:
                    line = input()
                    for sent in analyzer.analyze_line(line):
                        for ol in sent:
                            oc.write(ol)
        except EOFError:
            pass
        except KeyboardInterrupt:
            pass

    if parallel == 1:
        analyzer.set_nlp()
        for path in files:
            with output_context as oc:
                with open(path, "r") as f:
                    for line in f:
                        for sent in analyzer.analyze_line(line):
                            for ol in sent:
                                oc.write(ol)
    else:
        buffer = []
        for file_idx, path in enumerate(files):
            with open(path, "r") as f:
                while True:
                    eof, buffer = fill_buffer(f, MINI_BATCH_SIZE * parallel, buffer)
                    if eof and (file_idx + 1 < len(files) or len(buffer) == 0):
                        break  # continue to next file
                    if not pool:
                        if len(buffer) <= MINI_BATCH_SIZE:  # enough for single process
                            analyzer.set_nlp()
                            for line in buffer:
                                for sent in analyzer.analyze_line(line):
                                    for ol in sent:
                                        output_json_open()
                                        print(ol, file=output)
                            break  # continue to next file
                        parallel = (len(buffer) - 1) // MINI_BATCH_SIZE + 1
                        pool = Pool(parallel)

                    mini_batch_size = (len(buffer) - 1) // parallel + 1
                    mini_batches = [
                        buffer[idx * mini_batch_size : (idx + 1) * mini_batch_size] for idx in range(parallel)
                    ]
                    for mini_batch_result in pool.map(analyzer.analyze_lines_mp, mini_batches):
                        for sents in mini_batch_result:
                            for lines in sents:
                                for ol in lines:
                                    output_json_open()
                                    print(ol, file=output)

                    buffer.clear()  # process remaining part of current file


def analyze():
    pass


def analyze_parallel():
    pass


def fill_buffer(f: Iterator, batch_size: int, buffer: Optional[List[str]] = None):
    if buffer is None:
        buffer = []

    for line in f:
        buffer.append(line)
        if len(buffer) == batch_size:
            return False, buffer
    return True, buffer


@plac.annotations(
    model_path=("model directory path", "option", "b", str),
    split_mode=("split mode", "option", "s", str, ["A", "B", "C", None]),
    hash_comment=("hash comment", "option", "c", str, ["print", "skip", "analyze"]),
    output_path=("output path", "option", "o", Path),
    use_normalized_form=("overriding Token.lemma_ by normalized_form of SudachiPy", "flag", "n"),
    parallel=("parallel level (default=-1, all_cpus=0)", "option", "p", int),
    files=("input files", "positional"),
)
def run_ginzame(
    model_path=None,
    split_mode=None,
    hash_comment="print",
    output_path=None,
    use_normalized_form=False,
    parallel=-1,
    *files,
):
    run(
        model_path=model_path,
        ensure_model="ja_ginza",
        split_mode=split_mode,
        hash_comment=hash_comment,
        output_path=output_path,
        output_format="mecab",
        require_gpu=False,
        use_normalized_form=use_normalized_form,
        parallel=parallel,
        disable_sentencizer=False,
        files=files,
    )


def main_ginzame():
    plac.call(run_ginzame)


@plac.annotations(
    model_path=("model directory path", "option", "b", str),
    ensure_model=("select model either ja_ginza or ja_ginza_electra", "option", "m", str, ["ja_ginza", "ja-ginza", "ja_ginza_electra", "ja-ginza-electra", None]),
    split_mode=("split mode", "option", "s", str, ["A", "B", "C", None]),
    hash_comment=("hash comment", "option", "c", str, ["print", "skip", "analyze"]),
    output_path=("output path", "option", "o", Path),
    output_format=("output format", "option", "f", str, ["0", "conllu", "1", "cabocha", "2", "mecab", "3", "json"]),
    require_gpu=("enable require_gpu", "flag", "g"),
    use_normalized_form=("overriding Token.lemma_ by normalized_form of SudachiPy", "flag", "n"),
    disable_sentencizer=("disable spaCy's sentence separator", "flag", "d"),
    parallel=("parallel level (default=1, all_cpus=0)", "option", "p", int),
    files=("input files", "positional"),
)
def run_ginza(
    model_path=None,
    ensure_model=None,
    split_mode=None,
    hash_comment="print",
    output_path=None,
    output_format="conllu",
    require_gpu=False,
    use_normalized_form=False,
    disable_sentencizer=False,
    parallel=1,
    *files,
):
    run(
        model_path=model_path,
        ensure_model=ensure_model,
        split_mode=split_mode,
        hash_comment=hash_comment,
        output_path=output_path,
        output_format=output_format,
        require_gpu=require_gpu,
        use_normalized_form=use_normalized_form,
        disable_sentencizer=disable_sentencizer,
        parallel=parallel,
        files=files,
    )


def main_ginza():
    plac.call(run_ginza)


if __name__ == "__main__":
    plac.call(run_ginza)
