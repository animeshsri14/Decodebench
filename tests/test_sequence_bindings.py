import pytest

from decodebench.sequence import Sequence


def test_named_fanin_and_fanout_consumer_counts():
    seq = Sequence("dag")

    @seq.stage
    def source(x):
        return x

    @seq.stage
    def left(source):
        return source

    @seq.stage
    def right(source):
        return source

    @seq.stage
    def merge(left, right):
        return left

    assert seq._dependency_consumer_counts({"x": object()}) == {
        "source": 2,
        "left": 1,
        "right": 1,
    }


def test_unconsumed_output_fails_before_execution():
    seq = Sequence("bad")

    @seq.stage
    def first(x):
        return x

    @seq.stage
    def second(x):
        return x

    with pytest.raises(ValueError, match="first.*never consumed"):
        seq._dependency_consumer_counts({"x": object()})


def test_unsupported_signature_and_duplicate_names_fail_registration():
    seq = Sequence("signatures")

    with pytest.raises(TypeError, match="plain positional"):
        @seq.stage
        def bad(*args):
            return args

    def same(x):
        return x

    seq.stage(same)
    with pytest.raises(ValueError, match="duplicate stage"):
        seq.stage(same)
