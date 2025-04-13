import argparse
import logging
import os

def run_one_contract_problem():
    from one_contract import main as run
    run()

def run_two_contracts_problem():
    from two_contracts import main as run
    run()

def run_N_contracts_problem(N):
    from N_contracts import main as run
    run(N)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insurance contract optimization")

    parser.add_argument(
        "problem",
        choices=["one_contract_problem", "two_contracts_problem", "N_contracts_problem"],
        help="Choose the problem to solve",
    )

    parser.add_argument(
        "--N",
        type=int,
        help="Specify the number of contracts (N) for the N contracts problem",
        required=False
    )

    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging verbosity level"
    )

    args = parser.parse_args()

    #Setup logging BEFORE anything else uses it
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, args.loglevel),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/optimization.log", mode='w')
        ]
    )

    if args.problem == "one_contract_problem":
        run_one_contract_problem()
    elif args.problem == "two_contracts_problem":
        run_two_contracts_problem()
    elif args.problem == "N_contracts_problem":
        if args.N is None:
            logging.error("You must specify the number of contracts using the --N argument")
        else:
            run_N_contracts_problem(args.N)