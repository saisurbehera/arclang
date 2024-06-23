"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Arclang."""


if __name__ == "__main__":
    main(prog_name="arclang")  # pragma: no cover
