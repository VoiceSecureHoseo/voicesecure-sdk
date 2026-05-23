"""PGD chunked 테스트 — PGD 제거로 인해 전체 skip."""

import pytest

pytestmark = pytest.mark.skip(reason="PGD removed from pipeline")


def test_short_audio_uses_single_pgd():
    pass


def test_output_shape_matches_audio():
    pass


def test_masking_constraint_per_chunk():
    pass


def test_no_nan_in_output():
    pass
