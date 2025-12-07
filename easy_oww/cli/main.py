"""
Main CLI entry point for easy-oww
"""
import click
from rich.console import Console
from easy_oww import __version__
from easy_oww.cli import commands
from easy_oww.utils.logger import setup_logger

console = Console()
logger = setup_logger()


@click.group()
@click.version_option(version=__version__)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, verbose):
    """
    easy-oww: Simplified OpenWakeWord ONNX model creation tool

    This tool guides you through creating custom wake word models for OpenWakeWord.
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose

    if verbose:
        import logging
        logger.setLevel(logging.DEBUG)


@cli.command()
@click.option('--workspace', '-w', default=None, help='Custom workspace path')
@click.pass_context
def init(ctx, workspace):
    """Initialize easy-oww workspace"""
    commands.init_workspace(workspace, ctx.obj.get('verbose', False))


@cli.command()
@click.option('--required-only', '-r', is_flag=True, help='Download only required datasets')
@click.option('--workspace', '-w', default=None, help='Custom workspace path')
@click.pass_context
def download(ctx, required_only, workspace):
    """Download required datasets"""
    commands.download_datasets(workspace, required_only, ctx.obj.get('verbose', False))


@cli.command()
@click.argument('project_name')
@click.option('--workspace', '-w', default=None, help='Custom workspace path')
@click.option('--wake-word', help='Wake word/phrase')
@click.option('--samples', type=int, default=1000, help='Number of training samples')
@click.option('--steps', type=int, default=10000, help='Training steps')
@click.option('--duration', type=float, default=1.5, help='Recording duration in seconds (default: 1.5)')
@click.pass_context
def create(ctx, project_name, workspace, wake_word, samples, steps, duration):
    """Create new wake word project"""
    commands.create_project(
        project_name,
        workspace,
        wake_word,
        samples,
        steps,
        duration,
        ctx.obj.get('verbose', False)
    )


@cli.command()
@click.argument('project_name')
@click.option('--workspace', '-w', default=None, help='Custom workspace path')
@click.option('--count', '-c', type=int, default=20, help='Number of samples to record')
@click.option('--duration', type=float, default=1.5, help='Recording duration in seconds (default: 1.5)')
@click.pass_context
def record(ctx, project_name, workspace, count, duration):
    """Record wake word samples"""
    commands.record_samples(
        project_name,
        workspace,
        count,
        duration,
        ctx.obj.get('verbose', False)
    )


@cli.command()
@click.argument('project_name')
@click.option('--workspace', '-w', default=None, help='Custom workspace path')
@click.option('--resume', is_flag=True, help='Resume from last checkpoint')
@click.option('--force', '-f', is_flag=True, help='Force full retrain (regenerate all clips and features)')
@click.pass_context
def train(ctx, project_name, workspace, resume, force):
    """Train wake word model"""
    commands.train_model(
        project_name,
        workspace,
        resume,
        ctx.obj.get('verbose', False),
        force
    )


@cli.command()
@click.argument('project_name')
@click.option('--workspace', '-w', default=None, help='Custom workspace path')
@click.option('--duration', '-d', type=int, default=60, help='Test duration in seconds')
@click.pass_context
def test(ctx, project_name, workspace, duration):
    """Test trained model"""
    commands.test_model(
        project_name,
        workspace,
        duration,
        ctx.obj.get('verbose', False)
    )


@cli.command('list')
@click.option('--workspace', '-w', default=None, help='Custom workspace path')
def list_projects(workspace):
    """List all projects"""
    commands.list_projects(workspace)


@cli.command('download-voices')
@click.option('--workspace', '-w', default=None, help='Custom workspace path')
@click.option('--language', '-l', default='en_US', help='Language code (e.g., en_US, en_GB)')
@click.option('--count', '-c', type=int, default=2, help='Number of voices to download')
@click.pass_context
def download_voices_cmd(ctx, workspace, language, count):
    """Download TTS voice models"""
    commands.download_voices(workspace, language, count, ctx.obj.get('verbose', False))


@cli.command('list-voices')
@click.option('--workspace', '-w', default=None, help='Custom workspace path')
def list_voices_cmd(workspace):
    """List installed TTS voices"""
    commands.list_voices(workspace)


if __name__ == '__main__':
    cli()
