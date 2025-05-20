# third party
import click

# first party
from fmplug.tasks.super_resolution_v2 import super_resolution_task


@click.group()
def cli():  # noqa
    pass


@cli.command("run-super-resolution-task")
@click.option("--config_name")
def run_super_resolution_task(config_name: str) -> None:
    super_resolution_task(config_name=config_name)


if __name__ == "__main__":
    # Be able to run different commands
    cli()
