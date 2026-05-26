# third party
import click


@click.group()
def cli() -> None:
    """Scientific inverse-problem runners backed by InverseBench."""


def _run_scientific_task(config_name: str) -> None:
    from fmplug.tasks.fmplug_inversebench import solve

    solve(config_name=config_name)


@cli.command("run-FMPlug-scientific-task")
@click.option("--config_name", required=True, help="Config name from fmplug/configs without .yaml.")
def solve_scientific_inv_prob(config_name: str) -> None:
    _run_scientific_task(config_name=config_name)


@cli.command("run-FMPlug-task")
@click.option("--config_name", required=True, help="Deprecated alias for run-FMPlug-scientific-task.")
def solve_scientific_inv_prob_alias(config_name: str) -> None:
    _run_scientific_task(config_name=config_name)


if __name__ == "__main__":
    cli()
