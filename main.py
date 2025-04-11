import argparse

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

    # Adding the N argument only when the user selects the N_contracts_problem
    parser.add_argument(
        "--N",
        type=int,
        help="Specify the number of contracts (N) for the N contracts problem",
        required=False
    )

    args = parser.parse_args()

    if args.problem == "one_contract_problem":
        run_one_contract_problem()
    elif args.problem == "two_contracts_problem":
        run_two_contracts_problem()
    elif args.problem == "N_contracts_problem":
        if args.N is None:
            print("You must specify the number of contracts using the --N argument")
        else:
            run_N_contracts_problem(args.N)
