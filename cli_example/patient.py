#/usr/bin/env python
"""A click test module."""

import click


@click.command()
@click.argument('patient_id')
@click.option('--verbose', '-v', is_flag=True, help="Print more output.")
def pid(patient_id, verbose):
    """Displays information from a patient ID."""
    print(f"Patient ID {patient_id}, verbose {'on' if verbose else 'off'}")


@click.command()
@click.argument('m')
def another_function(m):
    """Another function's help, to test API."""
    print(f"Another function's print: {m}.")


if __name__ == "__main__":
    pid()
