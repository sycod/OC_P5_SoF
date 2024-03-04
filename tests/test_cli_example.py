from cli_example.patient import pid, another_function


@pytest.mark.parametrize("patient_id", ["42", "new_patient"])
@pytest.mark.parametrize("verbose", [True, False])
def test_pid(patient_id, verbose):
    """Test cli_example.patient pid function"""
    message = f"Patient ID {patient_id}, verbose {'on' if verbose else 'off'}"
    assert pid(patient_id, verbose) == message

def test_another_function():
    assert  another_function() == "Another function's print."