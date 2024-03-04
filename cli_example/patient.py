"""A click test module."""

import click


@click.command()
@click.argument('patient_id')
@click.option('--verbose', '-v', is_flag=True, help="Print more output.")
def pid(patient_id, verbose):
    """Displays information from a patient ID."""
    message = f"Patient ID {patient_id}, verbose {'on' if verbose else 'off'}"
    return message


def another_function():
    """Another function's help, to test API."""

    return "Another function's print."


if __name__ == "__main__":
    print(help(pid))
