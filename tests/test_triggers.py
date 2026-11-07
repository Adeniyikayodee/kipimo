import pytest

from kipimo.config import CONFIG_DIR
from kipimo.payout import triggers

GOOD = (CONFIG_DIR / "triggers.yaml").read_text()


@pytest.mark.parametrize(
    "bad",
    [
        GOOD.replace("pays: 0.50", "pays: 0.125", 1),  # not a whole percent
        GOOD.replace("pays: 0.20", "pays: 0.60", 1),  # pays falls as loss rises
        GOOD.replace("id: 2\n", "id: 1\n", 1),  # reused id
        GOOD.replace("rice:", "beans:"),  # crop missing
        GOOD + "maize:\n  - id: 99\n    if: loss_mean >= 0.1\n    pays: 0.05\n",  # duplicate key
        GOOD.replace("loss_mean >= 0.60", "loss_mean >= 1.5", 1),
    ],
)
def test_bad_rule_files_are_refused(tmp_path, bad):
    (tmp_path / "t.yaml").write_text(bad)
    with pytest.raises(ValueError):
        triggers.load(tmp_path / "t.yaml")


def test_rule_file_loads():
    rules, gate = triggers.load()
    assert gate == 0.8 and [r.trigger_id for r in rules["maize"]] == [1, 2, 3]
