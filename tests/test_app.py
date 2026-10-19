import joblib
import pytest
from streamlit.testing.v1 import AppTest

from asd import aq10, config
from asd.data import FEATURES, TARGET, load_dataset
from asd.model import AgeGroupEnsemble

APP = str(config.ROOT / "app.py")


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    import sklearn
    df = load_dataset(config.DATA_PATH)
    model = AgeGroupEnsemble(random_state=0).fit(df[FEATURES], df[TARGET])
    metrics = {"sklearn_version": sklearn.__version__, "n_test": 1,
               "test": {"accuracy": 1.0, "roc_auc": 1.0}}
    path = tmp_path_factory.mktemp("model") / "asd_model.joblib"
    joblib.dump({"model": model, "test_index": [], "metrics": metrics}, path)
    return path


@pytest.fixture
def app(model_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_PATH", model_path)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    return at


def answer(at, group, autistic: bool, skip: int | None = None):
    for i, item in enumerate(aq10.QUESTIONNAIRES[group], start=1):
        if i == skip:
            continue
        on_agree_side = item.agree_scores == autistic
        at.radio(key=f"{group}_q{i}").set_value("Definitely agree" if on_agree_side else "Definitely disagree")
    at.selectbox(key="gender").set_value("Male")
    at.selectbox(key="ethnicity").set_value("White-European")


def submit(at):
    at.button[0].click().run()
    assert not at.exception
    return at


def test_adult_high_score(app):
    answer(app, aq10.ADULT, autistic=True)
    submit(app)
    assert app.metric[0].value == "10 / 10"
    assert "Higher likelihood" in app.error[0].value


def test_adult_low_score(app):
    answer(app, aq10.ADULT, autistic=False)
    submit(app)
    assert app.metric[0].value == "0 / 10"
    assert "Lower likelihood" in app.success[0].value
    assert not app.error


def test_adolescent_uses_adolescent_questionnaire(app):
    app.number_input[0].set_value(14).run()
    assert aq10.QUESTIONNAIRES[aq10.ADOLESCENT][0].text in app.radio(key="Adolescent_q1").label
    answer(app, aq10.ADOLESCENT, autistic=True)
    submit(app)
    assert app.metric[0].value == "10 / 10"
    assert "Higher likelihood" in app.error[0].value


def test_unanswered_questions_are_reported(app):
    submit(app)
    message = app.error[0].value
    assert "Please answer everything" in message
    assert all(f"question {i}" in message for i in range(1, 11))
    assert "gender" in message and "ethnicity" in message
    assert not app.metric


def test_partially_answered_form_lists_only_missing_items(app):
    answer(app, aq10.ADULT, autistic=True, skip=3)
    submit(app)
    message = app.error[0].value
    assert "question 3" in message and "question 1," not in message
    assert not app.metric


def test_about_section_always_renders(app):
    assert app.expander
    submit(app)
    assert app.expander
