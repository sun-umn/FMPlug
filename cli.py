# third party
import click

# first party
# from fmplug.tasks.super_resolution_v1 import super_resolution_task
from fmplug.tasks.inverse_solver_template import super_resolution_task


@click.group()
def cli():  # noqa
    pass


@cli.command("run-super-resolution-task")
def run_super_resolution_task() -> None:
    super_resolution_task(config_name="MSI")


if __name__ == "__main__":
    # Be able to run different commands
    cli()
