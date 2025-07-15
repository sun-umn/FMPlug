# third party
import click

# first party
from fmplug.tasks.linear_task_ada_dgp_lora import solve


@click.group()
def cli():  # noqa
    pass


@cli.command("run-FMPlug-task")
@click.option("--config_name")
def solve_inv_prob(config_name: str) -> None:
    solve(config_name=config_name)


if __name__ == "__main__":
    # Be able to run different commands
    cli()
