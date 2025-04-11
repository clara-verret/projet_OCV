import numpy as np
from scipy.stats import beta
from matplotlib import pyplot as plt
from config import a, b, L, R_min, R_max
from two_contracts import profit_from_contract, A_segment, P_theta
from scipy.optimize import LinearConstraint, differential_evolution

def profit_n_contracts(params, N):
    """
    Calculate total profit from N contracts.

    Args:
        params (array): [θ1, ..., θ_{N-1}, R1, ..., RN]
        N (int): Number of contracts

    Returns:
        float: Total profit
    """
    # Extract and sort theta values properly
    thetas_internal = params[:N].copy()
    thetas = np.concatenate((thetas_internal, [1]))
    Rs = params[N:]
    # Validate parameter values
    if any(th <= 0 or th >= 1 for th in thetas_internal):
        return -np.inf

    # Check that thetas are in ascending order
    if not np.all(np.diff(thetas) > 0):
        return -np.inf

    total_profit = 0
    for i in range(N):
        θ_low = thetas[i]
        θ_high = thetas[i+1]
        R = Rs[i]
        total_profit += profit_from_contract(θ_low, θ_high, R)
    return total_profit

def optimize_n_contracts(N, x0=None):
    """
    Optimize N contracts.

    Args:
        N (int): Number of contracts
        x0 (array, optional): Initial guess for [θ1,...,θ_{N-1}, R1,...,R{N-1}]

    Returns:
        tuple: (result, output_dict)
    """
    if x0 is None:
        # Create better initial guesses with clear separation
        θ_guesses = np.linspace(0.1, 0.9, N)
        R_guesses = np.linspace(R_min, R_max, N)
        x0 = np.concatenate((θ_guesses, R_guesses))

    # Set appropriate bounds
    bounds = [(0.001, 0.999)] * N + [(R_min, R_max)] * N 

    # Constraint: θ1 < θ2
    A = []
    epsilon = 1e-4
    total_dim = 2*N

    for i in range(N - 1):
        row = [0] * total_dim
        row[i] = -1
        row[i + 1] = 1
        A.append(row)

    A = np.array(A)
    lb = np.full(A.shape[0], epsilon)
    ub = np.full(A.shape[0], np.inf)

    linear_constraint = LinearConstraint(A, lb, ub)

    result = differential_evolution(
        func = lambda x: -profit_n_contracts(x,N), 
        x0 = x0, 
        bounds=bounds,
        constraints=(linear_constraint,)
    )

    output = {}
    if result.success:
        thetas_internal = result.x[:N]
        thetas = np.concatenate((thetas_internal, [1]))
        Rs = result.x[N:]
        profits = []
        Ps = []
        Qs = []
        As = []
        for i in range(N):
            θ_low, θ_high = thetas[i], thetas[i+1]
            R = Rs[i]
            P = P_theta(θ_low, R)
            Q = beta.cdf(θ_high, a, b) - beta.cdf(θ_low, a, b)
            A = A_segment(θ_low, θ_high)
            profit = (P - R * A) * Q
            profits.append(profit)
            Ps.append(P)
            Qs.append(Q)
            As.append(A)

        output = {
            'thetas': thetas,
            'Rs': Rs,
            'Ps': Ps,
            'Qs': Qs,
            'As': As,
            'profits': profits,
            'total_profit': -result.fun
        }

        print(f"Optimal {N} Contracts:")
        for i in range(N):
            print(f"Contract {i+1}: θ ∈ [{thetas[i]:.4f}, {thetas[i+1]:.4f}]")
            print(f"  R = {Rs[i]:.4f}, P = {Ps[i]:.4f}, Q = {Qs[i]:.4f}, A = {As[i]:.4f}, Profit = {profits[i]:.4f}")
        print(f"Total Profit: {-result.fun:.4f}")
    else:
        print("Optimization failed:", result.message)
        print("Try with different initial values or optimization parameters.")

    return result, output

def plot_n_contracts(output_n, output_single=None):
    """
    Create visualizations for N-contract model.
    
    Args:
        output_n (dict): Results from N-contract optimization
        output_single (dict, optional): Results from single-contract optimization
    """
    if not output_n or 'thetas' not in output_n:
        print("No valid optimization results to plot")
        return
    
    thetas = output_n['thetas']
    Rs = output_n['Rs']
    Ps = output_n['Ps']
    Qs = output_n['Qs']
    As = output_n['As']
    profits = output_n['profits']
    N = len(Rs)

    plt.figure(figsize=(16, 10))

    # --- Plot 1: Beta distribution with thresholds ---
    plt.subplot(2, 2, 1)
    x_values = np.linspace(0, 1, 1000)
    plt.plot(x_values, beta.pdf(x_values, a, b), label='Beta PDF')

    colors = plt.cm.viridis(np.linspace(0, 1, N))

    for i in range(1, len(thetas)-1):
        plt.axvline(x=thetas[i], linestyle='--', color=colors[i-1], 
                    label=f'θ{i} = {thetas[i]:.4f}')

    for i in range(N):
        plt.fill_between(x_values, 0, beta.pdf(x_values, a, b),
                         where=((x_values >= thetas[i]) & (x_values < thetas[i+1])),
                         alpha=0.3, color=colors[i])

    if output_single:
        plt.axvline(x=output_single['theta'], color='black', linestyle=':',
                    label=f'Single θ = {output_single["theta"]:.4f}')
    
    plt.title(f'Beta({a},{b}) Distribution with Contract Segments')
    plt.xlabel('θ (risk probability)')
    plt.ylabel('Density')
    plt.legend()

    # --- Plot 2: WTP Curves ---
    plt.subplot(2, 2, 2)
    theta_vals = np.linspace(0.01, 0.99, 200)

    for i in range(N):
        wtp_vals = [P_theta(t, Rs[i]) for t in theta_vals]
        plt.plot(theta_vals, wtp_vals, label=f'Contract {i+1} (R={Rs[i]:.2f})', color=colors[i])
        plt.plot(thetas[i], Ps[i], 'o', color=colors[i], label=f'P{i+1}={Ps[i]:.4f}')

    if output_single:
        wtp_single = [P_theta(t, output_single['R']) for t in theta_vals]
        plt.plot(theta_vals, wtp_single, 'k--', label=f'Single (R={output_single["R"]:.2f})')
        plt.plot(output_single['theta'], output_single['P'], 'ko', label=f'Ps={output_single["P"]:.4f}')

    plt.title('Willingness-to-Pay Functions')
    plt.xlabel('θ (risk probability)')
    plt.ylabel('Premium')
    plt.legend()

    # --- Plot 3: Indemnity vs Risk ---
    plt.subplot(2, 2, 3)
    for i in range(N):
        plt.hlines(Rs[i], thetas[i], thetas[i+1], color=colors[i], linewidth=2,
                   label=f'R{i+1} = {Rs[i]:.2f}')
        plt.axvline(thetas[i], color=colors[i], linestyle='--', alpha=0.6)

    plt.axvline(thetas[-1], color=colors[-1], linestyle='--', alpha=0.6)

    if output_single:
        plt.hlines(output_single['R'], output_single['theta'], 1, color='black', linestyle=':', label=f'Single R = {output_single["R"]:.2f}')

    plt.title('Indemnity Levels by Risk Segment')
    plt.xlabel('θ (risk probability)')
    plt.ylabel('Indemnity (R)')
    plt.ylim(0, L)
    plt.legend()

    # --- Plot 4: Profit Breakdown ---
    plt.subplot(2, 2, 4)
    categories = [f'C{i+1}' for i in range(N)] + ['Total']
    profits_all = profits + [output_n['total_profit']]

    if output_single:
        categories.append('Single')
        profits_all.append(output_single['profit'])

    bar_colors = list(colors) + ['gray']
    if output_single:
        bar_colors.append('black')

    bars = plt.bar(categories, profits_all, color=bar_colors)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.001, f'{yval:.4f}', ha='center')

    plt.title('Profit Comparison')
    plt.ylabel('Profit')
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'./plots/N_{N}_insurance_contracts.png')


######################################## Main ########################################
def main(N):
    result, output = optimize_n_contracts(N)
    plot_n_contracts(output_n=output)

if __name__ == "__main__":
    main()