import pytest

from asd import aq10
from asd.aq10 import ADOLESCENT, ADULT, QUESTIONNAIRES, RESPONSES, age_group, score_answers

# Items scored on "agree" per the published AQ-10 scoring keys.
AGREE_ITEMS = {ADULT: {1, 7, 8, 10}, ADOLESCENT: {1, 5, 8, 10}}


@pytest.mark.parametrize("group", [ADULT, ADOLESCENT])
def test_questionnaire_has_ten_items_with_published_scoring_key(group):
    items = QUESTIONNAIRES[group]
    assert len(items) == 10
    assert {i for i, item in enumerate(items, 1) if item.agree_scores} == AGREE_ITEMS[group]


@pytest.mark.parametrize("group", [ADULT, ADOLESCENT])
@pytest.mark.parametrize("response", RESPONSES)
def test_scoring_direction(group, response):
    agreed = response in RESPONSES[:2]
    expected = [int(agreed == (i in AGREE_ITEMS[group])) for i in range(1, 11)]
    assert score_answers(group, [response] * 10) == expected


def test_all_autistic_side_answers_score_ten():
    responses = ["Definitely agree" if i in AGREE_ITEMS[ADULT] else "Definitely disagree" for i in range(1, 11)]
    assert sum(score_answers(ADULT, responses)) == 10


def test_score_answers_rejects_bad_input():
    with pytest.raises(ValueError):
        score_answers(ADULT, ["Definitely agree"] * 9)
    with pytest.raises(ValueError):
        score_answers(ADULT, ["Agree"] * 10)


@pytest.mark.parametrize("age, group", [(12, ADOLESCENT), (16, ADOLESCENT), (16.9, ADOLESCENT),
                                        (17, ADULT), (64, ADULT)])
def test_age_group(age, group):
    assert age_group(age) == group


def test_age_group_rejects_under_12():
    with pytest.raises(ValueError):
        age_group(aq10.ADOLESCENT_MIN_AGE - 1)
