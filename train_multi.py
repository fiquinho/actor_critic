import argparse
from subprocess import call
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Execute multiple experiments from configurations files "
                                                 "one after the other.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    required_named = parser.add_argument_group('REQUIRED named arguments')
    required_named.add_argument("--configs_dir", type=str, required=True,
                                help="All .json files in this folder must be accepted configurations files "
                                     "to train agents.")
    args = parser.parse_args()

    configs_dir = Path(args.configs_dir)

    for config in configs_dir.iterdir():

        if config.suffix != ".json":
            continue

        call_list = ["python", "train_agent.py",
                     "--config_file", str(config),
                     "--replace"]

        call(call_list)


if __name__ == '__main__':
    main()
