"""Test module"""

import pytest
from click.testing import CliRunner

from cli_example.patient import pid, another_function


@pytest.mark.parametrize("patient_id", ["42", "new_patient"])
def test_pid(patient_id):
# def test_pid():
    """Test cli_example.patient pid function"""
    # with & without verbose
    output = CliRunner().invoke(pid, [patient_id]).output
    message = f"Patient ID {patient_id}, verbose off\n"
    assert output == message

    output = CliRunner().invoke(pid, [patient_id, "-v"]).output
    message = f"Patient ID {patient_id}, verbose on\n"
    assert output == message


@pytest.mark.parametrize("m", ["1337", "another_argument"])
def test_another_function(m):
    """Test cli_example.patient other function"""
    output = CliRunner().invoke(another_function, [m]).output
    assert output == f"Another function's print: {m}.\n"
