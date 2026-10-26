from eval.scoring import is_refusal, score


def test_fixed_sentence_and_variants_are_refusals():
    assert is_refusal("I could not find this information in the patient's record.")
    assert is_refusal("Agueda Gerhold's NHS number is not listed in the patient's record.")
    assert is_refusal("The blood type is not recorded in the chart.")


def test_real_answers_are_not_refusals():
    assert not is_refusal("Atorvastatin 40 MG, started 2016.")
    assert not is_refusal("The most recent BMI was 28.8 kg/m2 on 2021-11-06.")


def test_numbers_check():
    ok, _ = score({"type": "numbers", "numbers": [28.8], "tol": 0.15}, "BMI 28.79 kg/m2")
    assert ok
    ok, _ = score({"type": "numbers", "numbers": [28.8], "tol": 0.15}, "BMI percentile 97.859 %")
    assert not ok
