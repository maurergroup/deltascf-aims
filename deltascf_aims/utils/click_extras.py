from click import Argument, Option, UsageError


def get_long_arg_name(ctx, arg):
    """
    Get the long argument name for a given argument.

    Parameters
    ----------
    ctx : click.Context
        The click context object
    arg : str
        The parsed argument name to get the long name for

    Returns
    -------
    str
        The long version of the CLI argument name
    """
    # Convert set back to string
    if isinstance(arg, set):
        arg = ", ".join(arg)

    # Necessary as ctx.command.params is a list of click.Option objects so we need to
    # extract the name attribute from each object
    arg_index = [i.name for i in ctx.command.params].index(arg)

    return ctx.command.params[arg_index].opts[1]


class ShowHelpSubCmd(Argument):
    """Short circuit to show help documentation if --help is passed in a subcommand."""

    def handle_parse_result(self, ctx, opts, args):
        # Check if '--help' exists on the command line
        if any(arg in ctx.help_option_names for arg in args):
            # If asking for help, check if we are in a subcommand
            for arg in opts.values():
                if arg in ctx.command.commands:
                    # Matches a subcommand name, and '--help' is present
                    args = [arg] + args

        return super().handle_parse_result(ctx, opts, args)


class MutuallyExclusive(Option):
    """
    Allow a click option to be mutually exclusive with another option.

    ...

    Attributes
    ----------
    mutually_exclusive : set
        A set of mutually exclusive options.
    name : str
        The name of the option.
    """

    def __init__(self, *args, **kwargs):
        self.mutually_exclusive = set(kwargs.pop("mutually_exclusive", []))
        help = kwargs.get("help", "")

        if self.mutually_exclusive:
            ex_str = ", ".join(self.mutually_exclusive)
            kwargs["help"] = f"{help} [mutually exclusive with `{ex_str}`]"

        super().__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        curr_arg_long = get_long_arg_name(ctx, self.name)
        mut_ex_arg_long = get_long_arg_name(ctx, self.mutually_exclusive)

        if self.mutually_exclusive.intersection(opts) and self.name in opts:
            raise UsageError(
                f"`{curr_arg_long}` is mutually exclusive with `{mut_ex_arg_long}`."
            )

        return super().handle_parse_result(ctx, opts, args)


class MutuallyInclusive(Option):
    """
    Allow a click option to be mutually inclusive with another option.

    ...

    Attributes
    ----------
    mutually_inclusive : set
        A set of mutually inclusive options.
    name : str
        The name of the option.
    """

    def __init__(self, *args, **kwargs):
        self.mutually_inclusive = set(kwargs.pop("mutually_inclusive", []))
        help = kwargs.get("help", "")

        if self.mutually_inclusive:
            ex_str = ", ".join(self.mutually_inclusive)
            kwargs["help"] = f" {help} [mutually inclusive with `{ex_str}`]"

        super().__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        curr_arg_long = get_long_arg_name(ctx, self.name)
        mut_ex_arg_long = get_long_arg_name(ctx, self.mutually_inclusive)

        if self.mutually_inclusive.intersection(opts) and self.name not in opts:
            msg = (
                f"`{curr_arg_long}` is mutually inclusive with arguments"
                f"`{mut_ex_arg_long}`."
            )
            raise UsageError(msg)

        return super().handle_parse_result(ctx, opts, args)
