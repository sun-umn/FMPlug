# third party
import click

# first party
from fmplug.tasks.non_linear_deblurring import non_linear_deblurring_task
from fmplug.tasks.super_resolution_v1 import super_resolution_task
from fmplug.tasks.turbulence import turbulence_task  # type: ignore


@click.group()
def cli():  # noqa
    pass


@cli.command("run-super-resolution-task")
@click.option("--config_name")
def run_super_resolution_task(config_name: str) -> None:
    super_resolution_task(config_name=config_name)


@cli.command("run-turbulence-task")
@click.option("--config_name")
def run_turbulence_task(config_name: str) -> None:
    turbulence_task(config_name=config_name)


@cli.command("run-non-linear-deblurring-task")
@click.option("--config_name")
def run_non_linear_deblurring_task(config_name: str) -> None:
    non_linear_deblurring_task(config_name=config_name)


if __name__ == "__main__":
    # Be able to run different commands
    cli()
