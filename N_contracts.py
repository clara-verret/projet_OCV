import numpy as np
from scipy.stats import beta
from matplotlib import pyplot as plt
from config import a, b, L, R_min, R_max
from two_contracts import profit_from_contract, A_segment, P_theta
from scipy.optimize import LinearConstraint, differential_evolution
import logging
import time

def profit_n_contracts(params, N):
    """Calculate total profit from N contracts."""
    # Print parameter values for debugging
    if np.any(np.isnan(params)) or np.any(np.isinf(params)):
        logging.warning(f"WARNING: Input params contain NaN/Inf: {params}")
        return -np.inf
    
    # Extract theta values
    thetas_internal = params[:N].copy()
    thetas = np.concatenate((thetas_internal, [1]))
    Rs = params[N:]
    
    # Validate parameter values
    if any(th <= 0 or th >= 1 for th in thetas_internal):
        logging.warning(f"WARNING: Invalid theta range: {thetas_internal}")
        return -np.inf

    # Check that thetas are in ascending order
    if not np.all(np.diff(thetas) > 0):
        logging.warning(f"WARNING: Thetas not in ascending order: {thetas}")
        return -np.inf
    
    total_profit = 0
    for i in range(N):
        θ_low = thetas[i]
        θ_high = thetas[i+1]
        R = Rs[i]
        
        # Check individual values before calculation
        if np.isnan(θ_low) or np.isnan(θ_high) or np.isnan(R):
            logging.warning(f"WARNING: NaN values in segment {i}: θ_low={θ_low}, θ_high={θ_high}, R={R}")
            return -np.inf
            
        segment_profit = profit_from_contract(θ_low, θ_high, R)
        
        # Check for NaN/Inf in result
        if np.isnan(segment_profit) or np.isinf(segment_profit):
            logging.warning(f"WARNING: Invalid profit for segment {i}: {segment_profit}")
            logging.warning(f"Parameters: θ_low={θ_low}, θ_high={θ_high}, R={R}")
            return -np.inf
            
        total_profit += segment_profit
    
    # Final check on total profit
    if np.isnan(total_profit) or np.isinf(total_profit):
        logging.warning(f"WARNING: Invalid total profit: {total_profit}")
        return -np.inf
        
    return total_profit

def optimize_n_contracts(N, x0=None):
    """Optimize N contracts."""
    if x0 is None:
        # Create initial guesses with clear separation
        θ_guesses = np.linspace(0.1, 0.9, N)
        R_guesses = np.linspace(R_min, R_max, N)
        x0 = np.concatenate((θ_guesses, R_guesses))
    
    # Add debugging wrapper around objective function
    call_count = [0]
    def debug_objective(x):
        call_count[0] += 1
        if call_count[0] % 100 == 0:
            logging.info(f"Optimization iteration {call_count[0]}")
            
        result = -profit_n_contracts(x, N)
        
        # Check for invalid result
        if np.isnan(result) or np.isinf(result):
            logging.error(f"Invalid result at iteration {call_count[0]}")
            logging.error(f"Parameters: {x}")
            
        return result
    
    # Set appropriate bounds
    bounds = [(0.01, 0.99)] * N + [(R_min, R_max)] * N 

    # Constraint: θ1 < θ2 < ... < θN-1 < 1
    A = []
    epsilon = 1e-4
    total_dim = 2*N

    for i in range(N - 1):
        row = [0] * total_dim
        row[i] = -1
        row[i + 1] = 1
        A.append(row)

    # Add explicit lower bound constraints for thetas
    for i in range(N):
        row = [0] * total_dim
        row[i] = 1  # θᵢ > 0.01
        A.append(row)

    # Add explicit upper bound constraints for thetas
    for i in range(N):
        row = [0] * total_dim
        row[i] = -1  # -θᵢ > -0.99 (equivalent to θᵢ < 0.99)
        A.append(row)

    A = np.array(A)
    
    # Set lower bounds for all constraints
    lb = np.zeros(A.shape[0])
    lb[:N-1] = epsilon  # For ordering constraints
    lb[N-1:2*N-1] = 0.01  # Lower bounds for thetas
    lb[2*N-1:] = -0.99  # Upper bounds for thetas (expressed as lower bounds)
    
    ub = np.full(A.shape[0], np.inf)

    linear_constraint = LinearConstraint(A, lb, ub)

    # Try-except to catch optimization errors
    try:
        result = differential_evolution(
            func=debug_objective, 
            bounds=bounds,
            constraints=(linear_constraint,),
            maxiter=1000,
            popsize=15,
            tol=1e-5,
            updating='deferred',  # Try different updating schemes
        )
    except Exception as e:
        logging.error(f"Optimization error: {e}")
        # Create a mock result object for error cases
        from types import SimpleNamespace
        result = SimpleNamespace(success=False, x=x0, fun=np.inf, message=str(e))

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

############################## Plotting ############################

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
    start_time = time.time()
    logging.info(f"Starting optimization for {N} contracts...")
    
    result, output = optimize_n_contracts(N)
    plot_n_contracts(output_n=output)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Optimization and plotting completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()