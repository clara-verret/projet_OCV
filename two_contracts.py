import numpy as np
from scipy.stats import beta
from scipy.integrate import quad
from matplotlib import pyplot as plt
from config import a, b, L, lambda_, R_min, R_max
from scipy.optimize import LinearConstraint, differential_evolution


def P_theta(theta, R):
    """
    Calculate willingness-to-pay for a given risk probability theta and indemnity R
    
    Args:
        theta (float): Risk probability
        R (float): Indemnity amount
        
    Returns:
        float: Willingness to pay
    """
    if theta <= 0 or theta >= 1 or R <= 0 or R > L:
        return 0  # handle out of bounds
        
    num = 1 - theta + theta * np.exp(lambda_ * L)
    den = 1 - theta + theta * np.exp(lambda_ * (L - R))
    return (1 / lambda_) * np.log(num / den)

def A_segment(theta_low, theta_high):
    """
    Calculate average damage probability for a segment [θ_low, θ_high]
    
    Args:
        theta_low (float): Lower threshold
        theta_high (float): Upper threshold
        
    Returns:
        float: Average damage probability
    """
    if theta_low >= theta_high:
        return 0
        
    # Ensure bounds are valid for integration
    theta_low = max(0, min(theta_low, 1))
    theta_high = max(0, min(theta_high, 1))
    
    # If they're the same after bounding, return 0
    if abs(theta_high - theta_low) < 1e-10:
        return 0
    
    numerator, _ = quad(lambda x: x * beta.pdf(x, a, b), theta_low, theta_high)
    Q = beta.cdf(theta_high, a, b) - beta.cdf(theta_low, a, b)
    return numerator / Q if Q > 0 else 0

def profit_from_contract(theta_low, theta_high, R):
    """
    Calculate profit from a contract with segment [θ_low, θ_high] and indemnity R
    
    Args:
        theta_low (float): Lower risk threshold
        theta_high (float): Upper risk threshold
        R (float): Indemnity amount
        
    Returns:
        float: Profit from the contract
    """
    # Ensure bounds are valid
    theta_low = max(0, min(theta_low, 1))
    theta_high = max(0, min(theta_high, 1))
    
    if theta_low >= theta_high or R <= 0 or R > L:
        return 0
    
    # Premium determined by lower threshold
    P = P_theta(theta_low, R)
    
    # Quantity insured in this segment
    Q = beta.cdf(theta_high, a, b) - beta.cdf(theta_low, a, b)
    
    # Average damage probability in this segment
    A = A_segment(theta_low, theta_high)
    
    # Profit from this contract
    return (P - R * A) * Q

# Profit function for two contracts
def profit_two_contracts(params):
    """
    Calculate total profit from two contracts
    
    Args:
        params (array): [θ1, θ2, R1, R2] where θ1 < θ2 defines contract segments
        
    Returns:
        float: Total profit from both contracts
    """
    θ1, θ2, R1, R2 = params
    
    # Enforce ordering constraint with penalty
    if θ1 >= θ2:
        return -np.inf
    
    # Calculate profit from each contract
    profit1 = profit_from_contract(θ1, θ2, R1)
    profit2 = profit_from_contract(θ2, 1, R2)
    return profit1 + profit2

# Optimization
def optimize_two_contracts(x0=None):
    """
    Find optimal parameters for two contracts
    
    Args:
        x0 (array, optional): Initial guess [θ1, θ2, R1, R2]
        
    Returns:
        tuple: (result object, formatted output dict)
    """
    # Default initial guess if none provided
    if x0 is None:
        x0 = [0.1, 0.9, 0.1, 1.1]
    
    # Bounds: only constrain values to be within reasonable ranges
    bounds = [
        (0.001, 0.999),   # θ1
        (0.001, 0.999),   # θ2
        (R_min, R_max),   # R1
        (R_min, R_max)    # R2
    ]
    
    # Constraint: θ1 < θ2
    A = np.array([[-1, 1, 0, 0]])  # θ2 - θ1 ≥ ε
    lb = [1e-4]
    ub = [np.inf]
    linear_constraint = LinearConstraint(A, lb, ub)

    result = differential_evolution(
        func = lambda x: -profit_two_contracts(x), 
        x0 = x0, 
        bounds=bounds,
        constraints=(linear_constraint,)
    )
    
    output = {}
    
    if result.success:
        θ1_opt, θ2_opt, R1_opt, R2_opt = result.x
        P1_opt = P_theta(θ1_opt, R1_opt)
        P2_opt = P_theta(θ2_opt, R2_opt)
        
        # Calculate segment quantities
        Q1 = beta.cdf(θ2_opt, a, b) - beta.cdf(θ1_opt, a, b)
        Q2 = 1 - beta.cdf(θ2_opt, a, b)
        
        # Calculate average damage probabilities
        A1 = A_segment(θ1_opt, θ2_opt)
        A2 = A_segment(θ2_opt, 1)
        
        # Calculate profits
        profit1 = (P1_opt - R1_opt * A1) * Q1
        profit2 = (P2_opt - R2_opt * A2) * Q2
        
        output = {
            'theta1': θ1_opt,
            'theta2': θ2_opt,
            'R1': R1_opt,
            'R2': R2_opt,
            'P1': P1_opt,
            'P2': P2_opt,
            'Q1': Q1,
            'Q2': Q2,
            'A1': A1,
            'A2': A2,
            'profit1': profit1,
            'profit2': profit2,
            'total_profit': -result.fun
        }
        
        print("Optimal Two Contracts:")
        print(f"Contract 1: θ ∈ [{θ1_opt:.4f}, {θ2_opt:.4f}]")
        print(f"  R1 = {R1_opt:.4f}, P1 = {P1_opt:.4f}")
        print(f"  Q1 = {Q1:.4f}, A1 = {A1:.4f}, Profit1 = {profit1:.4f}")
        print(f"Contract 2: θ ∈ [{θ2_opt:.4f}, 1.0000]")
        print(f"  R2 = {R2_opt:.4f}, P2 = {P2_opt:.4f}")
        print(f"  Q2 = {Q2:.4f}, A2 = {A2:.4f}, Profit2 = {profit2:.4f}")
        print(f"Total Profit: {-result.fun:.4f}")
    else:
        print("Optimization failed:", result.message)
    
    return result, output

######################################## Plotting ##########################################
def plot_two_contracts(output_two, output_single=None):
    """
    Create visualizations for the two-contract model
    
    Args:
        output_two (dict): Results from two-contract optimization
        output_single (dict, optional): Results from single-contract optimization
    """
    if not output_two:
        print("No valid optimization results to plot")
        return
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Beta distribution with thresholds
    plt.subplot(2, 2, 1)
    x_values = np.linspace(0, 1, 1000)
    plt.plot(x_values, beta.pdf(x_values, a, b))
    
    # Add thresholds and fill regions
    plt.axvline(x=output_two['theta1'], color='r', linestyle='--', 
                label=f'θ1 = {output_two["theta1"]:.4f}')
    plt.axvline(x=output_two['theta2'], color='g', linestyle='--', 
                label=f'θ2 = {output_two["theta2"]:.4f}')
    
    # Fill Contract 1 region
    plt.fill_between(x_values, 0, beta.pdf(x_values, a, b), 
                    where=((x_values >= output_two['theta1']) & (x_values < output_two['theta2'])), 
                    alpha=0.3, color='red')
    
    # Fill Contract 2 region
    plt.fill_between(x_values, 0, beta.pdf(x_values, a, b), 
                    where=(x_values >= output_two['theta2']), 
                    alpha=0.3, color='green')
    
    if output_single:
        plt.axvline(x=output_single['theta'], color='blue', linestyle=':',
                   label=f'Single θ = {output_single["theta"]:.4f}')
    
    plt.title(f'Beta({a},{b}) distribution with contract thresholds')
    plt.xlabel('θ (risk probability)')
    plt.ylabel('Density')
    plt.legend()

    # Plot 2: WTP curve with contract details
    plt.subplot(2, 2, 2)
    theta_values = np.linspace(0.01, 0.99, 100)
    
    # WTP curves for both contracts
    wtp_values_1 = [P_theta(t, output_two['R1']) for t in theta_values]
    wtp_values_2 = [P_theta(t, output_two['R2']) for t in theta_values]
    
    plt.plot(theta_values, wtp_values_1, 'r-', label=f'Contract 1 (R={output_two["R1"]:.2f})')
    plt.plot(theta_values, wtp_values_2, 'g-', label=f'Contract 2 (R={output_two["R2"]:.2f})')
    
    # Highlight premium points
    plt.plot(output_two['theta1'], output_two['P1'], 'ro', 
             label=f'P1={output_two["P1"]:.4f}')
    plt.plot(output_two['theta2'], output_two['P2'], 'go', 
             label=f'P2={output_two["P2"]:.4f}')
    
    # Mark thresholds
    plt.axvline(x=output_two['theta1'], color='r', linestyle='--')
    plt.axvline(x=output_two['theta2'], color='g', linestyle='--')
    
    if output_single:
        wtp_values_single = [P_theta(t, output_single['R']) for t in theta_values]
        plt.plot(theta_values, wtp_values_single, 'b--', 
                label=f'Single (R={output_single["R"]:.2f})')
        plt.plot(output_single['theta'], output_single['P'], 'bo', 
                label=f'Ps={output_single["P"]:.4f}')
    
    plt.title('Willingness-to-Pay Functions')
    plt.xlabel('θ (risk probability)')
    plt.ylabel('Premium')
    plt.legend()

    # Plot 3: Indemnity vs Risk
    plt.subplot(2, 2, 3)
    
    # Draw horizontal lines for indemnity levels
    plt.hlines(output_two['R1'], 0, output_two['theta2'], color='r', 
              linestyle='-', label=f'R1 = {output_two["R1"]:.4f}')
    plt.hlines(output_two['R2'], output_two['theta2'], 1, color='g', 
              linestyle='-', label=f'R2 = {output_two["R2"]:.4f}')
    
    # Mark thresholds
    plt.axvline(x=output_two['theta1'], color='r', linestyle='--')
    plt.axvline(x=output_two['theta2'], color='g', linestyle='--')
    
    if output_single:
        plt.hlines(output_single['R'], output_single['theta'], 1, 
                  color='blue', linestyle=':', label=f'Single R = {output_single["R"]:.4f}')
    
    plt.title('Indemnity Levels by Risk Segment')
    plt.xlabel('θ (risk probability)')
    plt.ylabel('Indemnity (R)')
    plt.ylim(0, L)
    plt.legend()

    # Plot 4: Profit breakdown
    plt.subplot(2, 2, 4)
    
    categories = ['Contract 1', 'Contract 2', 'Total']
    profits = [output_two['profit1'], output_two['profit2'], output_two['total_profit']]
    
    if output_single:
        categories.append('Single Contract')
        profits.append(output_single['profit'])
    
    plt.bar(categories, profits)
    
    for i, v in enumerate(profits):
        plt.text(i, v + 0.001, f'{v:.4f}', ha='center')
    
    plt.title('Profit Comparison')
    plt.ylabel('Profit')
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('./plots/two_insurance_contract.png')

######################################## Main ########################################
def main():
    result = optimize_two_contracts()
    plot_two_contracts(result[1])

if __name__ == "__main__":
    main()