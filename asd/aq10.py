"""AQ-10 screening questionnaires (Allison, Auyeung & Baron-Cohen, 2012; used in NICE CG142).

Each item is answered on a 4-point scale and scores 1 point when the answer is on the
autistic-trait side. For some items that side is "agree", for others it is "disagree",
so the scoring direction is stored per item. The dataset's a1_score..a10_score columns
hold these item scores (not raw agree/disagree answers), in questionnaire order.
"""
from dataclasses import dataclass

ADULT = "Adult"
ADOLESCENT = "Adolescent"
GROUPS = (ADULT, ADOLESCENT)

# Adolescents are 12-16 and adults 17+ in the training data (18 adult rows are aged 17
# despite their age_desc saying "18 and more").
ADOLESCENT_MIN_AGE = 12
ADULT_MIN_AGE = 17

# A total above this suggests referral for a specialist assessment. The dataset's
# asd_status label is exactly this rule applied to the item scores.
REFERRAL_CUTOFF = 6

RESPONSES = ("Definitely agree", "Slightly agree", "Slightly disagree", "Definitely disagree")
_AGREE = RESPONSES[:2]


@dataclass(frozen=True)
class Item:
    text: str
    agree_scores: bool  # True: agreeing scores the point; False: disagreeing does


QUESTIONNAIRES = {
    # Self-report. Published for ages 16+; used here for 17+ to match the training data.
    ADULT: (
        Item("I often notice small sounds when others do not.", True),
        Item("I usually concentrate more on the whole picture, rather than the small details.", False),
        Item("I find it easy to do more than one thing at once.", False),
        Item("If there is an interruption, I can switch back to what I was doing very quickly.", False),
        Item("I find it easy to 'read between the lines' when someone is talking to me.", False),
        Item("I know how to tell if someone listening to me is getting bored.", False),
        Item("When I'm reading a story I find it difficult to work out the characters' intentions.", True),
        Item("I like to collect information about categories of things "
             "(e.g. types of car, types of bird, types of train, types of plant, etc.).", True),
        Item("I find it easy to work out what someone is thinking or feeling just by looking at their face.", False),
        Item("I find it difficult to work out people's intentions.", True),
    ),
    # Parent-report. Published for ages 12-15; used here for 12-16 to match the training data.
    ADOLESCENT: (
        Item("S/he notices patterns in things all the time.", True),
        Item("S/he usually concentrates more on the whole picture, rather than the small details.", False),
        Item("In a social group, s/he can easily keep track of several different people's conversations.", False),
        Item("If there is an interruption, s/he can switch back to what s/he was doing very quickly.", False),
        Item("S/he frequently finds that s/he doesn't know how to keep a conversation going.", True),
        Item("S/he is good at social chit-chat.", False),
        Item("When s/he was younger, s/he used to enjoy playing games involving pretending with other children.", False),
        Item("S/he finds it difficult to imagine what it would be like to be someone else.", True),
        Item("S/he finds social situations easy.", False),
        Item("S/he finds it hard to make new friends.", True),
    ),
}


def age_group(age: float) -> str:
    """Return the questionnaire / model group for an age, matching the training data split."""
    if age < ADOLESCENT_MIN_AGE:
        raise ValueError(f"AQ-10 screening here is for ages {ADOLESCENT_MIN_AGE}+, got {age}")
    return ADOLESCENT if age < ADULT_MIN_AGE else ADULT


def score_item(item: Item, response: str) -> int:
    if response not in RESPONSES:
        raise ValueError(f"Unknown response {response!r}; expected one of {RESPONSES}")
    agreed = response in _AGREE
    return int(agreed == item.agree_scores)


def score_answers(group: str, responses: list[str]) -> list[int]:
    """Convert the 10 raw responses for a group's questionnaire into a1..a10 item scores."""
    items = QUESTIONNAIRES[group]
    if len(responses) != len(items):
        raise ValueError(f"Expected {len(items)} responses, got {len(responses)}")
    return [score_item(item, r) for item, r in zip(items, responses)]
