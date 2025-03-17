from typing import Optional, Tuple, Union, Iterable
import sys

from paddlenlp.trl import ModelConfig
from paddlenlp.trl.utils import ScriptArguments
from paddlenlp.trainer.argparser import PdArgumentParser,DataClassType,DataClass


class TrlParser(PdArgumentParser):
    def __init__(
        self,
        dataclass_types: Optional[Union[DataClassType, Iterable[DataClassType]]] = None,
        **kwargs,
    ):
        # Make sure dataclass_types is an iterable
        if dataclass_types is None:
            dataclass_types = []
        elif not isinstance(dataclass_types, Iterable):
            dataclass_types = [dataclass_types]

        # TODO: 
        # Check that none of the dataclasses have the "config" field
        # for dataclass_type in dataclass_types:
        #     if "config" in dataclass_type.__dataclass_fields__:
        #         raise ValueError(
        #             f"Dataclass {dataclass_type.__name__} has a field named 'config'. This field is reserved for the "
        #             f"config file path and should not be used in the dataclass."
        #         )

        super().__init__(dataclass_types=dataclass_types, **kwargs)

    def parse_args_and_config(
        self, args: Optional[Iterable[str]] = None, return_remaining_strings: bool = False
    ) -> tuple[DataClass, ...]:
        args = list(args) if args is not None else sys.argv[1:]
        if "--config" in args:
            # Get the config file path from
            config_index = args.index("--config")
            args.pop(config_index)  # remove the --config flag
            config_path = args.pop(config_index)  # get the path to the config file
            with open(config_path) as yaml_file:
                config = yaml.safe_load(yaml_file)

            # Set the environment variables specified in the config file
            if "env" in config:
                env_vars = config.pop("env", {})
                if not isinstance(env_vars, dict):
                    raise ValueError("`env` field should be a dict in the YAML file.")
                for key, value in env_vars.items():
                    os.environ[key] = str(value)

            # Set the defaults from the config values
            config_remaining_strings = self.set_defaults_with_config(**config)
        else:
            config_remaining_strings = []

        # Parse the arguments from the command line
        output = self.parse_args_into_dataclasses(args=args, return_remaining_strings=return_remaining_strings)

        # Merge remaining strings from the config file with the remaining strings from the command line
        if return_remaining_strings:
            args_remaining_strings = output[-1]
            return output[:-1] + (config_remaining_strings + args_remaining_strings,)
        else:
            return output

    def set_defaults_with_config(self, **kwargs) -> list[str]:
        """
        Overrides the parser's default values with those provided via keyword arguments.

        Any argument with an updated default will also be marked as not required
        if it was previously required.

        Returns a list of strings that were not consumed by the parser.
        """
        # If an argument is in the kwargs, update its default and set it as not required
        for action in self._actions:
            if action.dest in kwargs:
                action.default = kwargs.pop(action.dest)
                action.required = False
        remaining_strings = [item for key, value in kwargs.items() for item in [f"--{key}", str(value)]]
        return remaining_strings