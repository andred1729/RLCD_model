import pytest
import torch

from rlcd.data import Collator, Example, load_jsonl, save_jsonl
from rlcd.synthetic import make_synthetic
from rlcd.tokenizer import ByteTokenizer, build_tokenizer, encode_batch


def test_example_validation():
    with pytest.raises(ValueError):
        Example("q", ["only one"], 0)
    with pytest.raises(ValueError):
        Example("q", ["a", "b"], 2)


def test_jsonl_roundtrip(tmp_path):
    ex = make_synthetic(20, seed=1)
    save_jsonl(ex, tmp_path / "x.jsonl")
    assert load_jsonl(tmp_path / "x.jsonl") == ex


def test_jsonl_error_reports_line(tmp_path):
    (tmp_path / "bad.jsonl").write_text('{"prompt": "q", "options": ["a", "b"], "label": 5}\n')
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        load_jsonl(tmp_path / "bad.jsonl")


def test_byte_tokenizer_truncates_to_max_length():
    tok = ByteTokenizer()
    ids = tok.encode_pair("p" * 500, "o" * 500, max_length=32)
    assert len(ids) <= 32 and ids[0] == tok.bos_id and tok.sep_id in ids


@pytest.mark.parametrize("prompt, option", [("q", "a" * 40 + "X"), ("", "a" * 61 + "X"), ("é", "α" * 20)])
def test_byte_tokenizer_preserves_pairs_that_fit(prompt, option):
    tok = ByteTokenizer()
    ids = tok.encode_pair(prompt, option, max_length=64)
    sep = ids.index(tok.sep_id)
    assert bytes(i - tok._offset for i in ids[1:sep]).decode("utf-8") == prompt
    assert bytes(i - tok._offset for i in ids[sep + 1:]).decode("utf-8") == option


def test_byte_tokenizer_fills_unused_prompt_budget():
    tok = ByteTokenizer()
    ids = tok.encode_pair("q", "a" * 100, max_length=64)
    assert len(ids) == 64
    assert ids[1:3] == [ord("q") + tok._offset, tok.sep_id]


def test_collate_shapes_and_padding():
    exs = [Example("a", ["x", "y"], 1), Example("bb", ["x", "y", "zzz"], 2, True, "s2")]
    b = Collator(build_tokenizer("byte"), 32)(exs)
    assert b["input_ids"].shape[:2] == (2, 3)
    assert b["option_mask"].tolist() == [[True, True, False], [True, True, True]]
    assert b["attention_mask"][0, 2].sum() == 0  # padded option slot is fully masked
    assert b["label"].tolist() == [1, 2] and b["ordinal"].tolist() == [False, True]
    assert b["source"] == ["default", "s2"]


def test_encode_batch_ids_within_vocab():
    tok = ByteTokenizer()
    b = encode_batch(tok, ["héllo ✓"], [["α", "b"]], 32)
    assert int(b["input_ids"].max()) < tok.vocab_size
    assert b["input_ids"].dtype == torch.long
