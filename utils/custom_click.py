from click import Option, UsageError


class MutuallyExclusive(Option):
    """Allow a click option to be mutually exclusive with another option."""

    def __init__(self, *args, **kwargs):
        self.mutually_exclusive = set(kwargs.pop("mutually_exclusive", []))
        help = kwargs.get("help", "")

        if self.mutually_exclusive:
            ex_str = ", ".join(self.mutually_exclusive)
            kwargs["help"] = help + (
                " | NOTE: This argument is mutually exclusive with: [" + ex_str + "]."
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
    """Allow a click option to be mutually inclusive with another option."""

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
