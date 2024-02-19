from click import Argument, Option, UsageError


class ShowHelpSubCmd(Argument):
    """
    Enable the help documentation to be shown for subcommands if '--help' is specified
    as part of the subcommand's args before any code is executed for the parent cmd.

    ...

    Methods
    -------
        handle_parse_result(ctx, opts, args)
            Check if '--help' exists on the command line and if we are in a subcommand.
    """

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

    Methods
    -------
        handle_parse_result(ctx, opts, args)
            Check if the mutually exclusive options are present in the command line
            arguments.
    """

    def __init__(self, *args, **kwargs):
        self.mutually_exclusive = set(kwargs.pop("mutually_exclusive", []))
        help = kwargs.get("help", "")

        if self.mutually_exclusive:
            ex_str = ", ".join(self.mutually_exclusive)
            kwargs["help"] = (
                f"{help} | NOTE: This argument is mutually exclusive with: [{ex_str}]."
            )

        super().__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        if self.mutually_exclusive.intersection(opts) and self.name in opts:
            raise UsageError(
                "Illegal usage: `{}` is mutually exclusive with "
                "arguments `{}`.".format(self.name, ", ".join(self.mutually_exclusive))
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

    Methods
    -------
        handle_parse_result(ctx, opts, args)
            Check if the mutually inclusive options are present in the command line
            arguments.
    """

    def __init__(self, *args, **kwargs):
        self.mutually_inclusive = set(kwargs.pop("mutually_inclusive", []))
        help = kwargs.get("help", "")

        if self.mutually_inclusive:
            ex_str = ", ".join(self.mutually_inclusive)
            kwargs["help"] = help + (
                " | NOTE: This argument is mutually inclusive with: [" + ex_str + "]."
            )

        super().__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        if self.mutually_inclusive.intersection(opts) and self.name not in opts:
            raise UsageError(
                "Illegal usage: `{}` is mutually inclusive with "
                "arguments `{}`.".format(self.name, ", ".join(self.mutually_inclusive))
            )

        return super().handle_parse_result(ctx, opts, args)
