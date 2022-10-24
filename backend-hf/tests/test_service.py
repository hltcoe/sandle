from datetime import timedelta

import hypothesis.strategies as st
import requests
from hypothesis import example, given, settings
from pytest import mark

PREDICT_URL = 'http://127.0.0.1:8000/v1/completions'
MODEL = 'facebook/opt-125m'


@settings(deadline=timedelta(seconds=15))
@given(prompt=st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126)))
@example(prompt="Mother's here.")
@example(prompt="Mother 's here.")
@mark.parametrize('trial', range(3))
def test_better_service_properties(prompt, trial):
    # Send input corpus to backend service at PREDICT_URL and parse response
    response = requests.post(PREDICT_URL, json=dict(model=MODEL, prompt=prompt))
    try:
        response.raise_for_status()
    except requests.HTTPError as orig_ex:
        # If response included an error message, include it in the test output
        try:
            error_message = str(response.json()['error'])
        except Exception:
            raise orig_ex
        else:
            raise requests.HTTPError(error_message, response=response)
    output = response.json()
    output_choices = output.get('choices')
    assert output_choices
    for output_choice in output_choices:
        assert output_choice['text']
