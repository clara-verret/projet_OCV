import numpy as np
from scipy.stats import beta
from scipy.optimize import minimize
from scipy.integrate import quad
from config import a, b, L, lambda_, R_min, R_max
from matplotlib import pyplot as plt

def P_theta(theta, R):
    if theta <= 0 or theta >= 1:
        return 0  # handle out of bounds
    num = 1 - theta + theta * np.exp(lambda_ * L)
    den = 1 - theta + theta * np.exp(lambda_ * (L - R))
    return (1 / lambda_) * np.log(num / den)

def A_theta(theta):
    """Average probability of damage for types in [θ, 1]"""
    Q = 1 - beta.cdf(theta, a, b)
    if Q <= 0:
        return 0
    numerator, _ = quad(lambda x: x * beta.pdf(x, a, b), theta, 1)
    return numerator / Q

def Q_theta(theta):
    """Proportion of population insured above threshold θ"""
    return 1 - beta.cdf(theta, a, b)

def profit(theta, R):
    if theta <= 0 or theta >= 1 or R <= 0 or R > L:
        return -np.inf  # invalid values
    P = P_theta(theta, R)
    A = A_theta(theta)
    Q = Q_theta(theta)
    return (P - R * A) * Q

def negative_profit(x):
    theta, R = x
    return -profit(theta, R)

############################ Optimization ############################

def optimize_contract():
    # Initial guess: (theta, R)
    x0 = [0.3, 0.5]

    # Bounds: θ ∈ (0.01, 0.99), R ∈ (0.01, L)
    bounds = [(0.01, 0.99), (0.01, L)]

    result = minimize(negative_profit, x0, method='L-BFGS-B', bounds=bounds)

    if result.success:
        theta_opt, R_opt = result.x
        P_opt = P_theta(theta_opt, R_opt)
        Q_opt = Q_theta(theta_opt)
        A_opt = A_theta(theta_opt)
        profit_opt = profit(theta_opt, R_opt)

        print(f"Successful optimization:")
        print(f"  - θ*      = {theta_opt:.4f}")
        print(f"  - R*      = {R_opt:.4f}")
        print(f"  - P*      = {P_opt:.4f}")
        print(f"  - Q (take-up rate) = {Q_opt:.4f}")
        print(f"  - A (avg damage)  = {A_opt:.4f}")
        print(f"  - Profit max      = {profit_opt:.4f}")
        return result.x, profit_opt
    else:
        print("Failed optimization:", result.message)
        return None, None

def generate_profit_curve(theta_opt=None, num_points=100):
    """
    Generate profit curve for various indemnity values with fixed optimal theta
    
    Returns:
    - R_values: Array of indemnity values
    - profit_values: Array of profit values
    """
    R_values = np.linspace(R_min, R_max, num_points)
    profit_values = []
    
    for R in R_values:
        if theta_opt is not None:
            # Use fixed optimal theta
            profit_values.append(profit(theta_opt, R))
        else:
            # For each R, find optimal theta
            bounds = [(0.01, 0.99)]
            result = minimize(lambda t: -profit(t[0], R), [0.3], method='L-BFGS-B', bounds=bounds)
            if result.success:
                profit_values.append(profit(result.x[0], R))
            else:
                profit_values.append(np.nan)
    
    return R_values, profit_values

############################## Plotting ############################

def generate_wtp_curve(theta_min=0.01, theta_max=0.99, R_opt=None, num_points=100):
    """
    Generate WTP curve for various theta values with fixed optimal R
    
    Returns:
    - theta_values: Array of theta values
    - wtp_values: Array of willingness-to-pay values
    """
    if R_opt is None:
        return [], []
    
    theta_values = np.linspace(theta_min, theta_max, num_points)
    wtp_values = [P_theta(t, R_opt) for t in theta_values]
    
    return theta_values, wtp_values

def plot_insurance_model():
    """
    Plot the insurance model results using the optimization approach
    """
    # Get optimal contract parameters
    (theta_opt, R_opt), _ = optimize_contract()
    
    if theta_opt is None or R_opt is None:
        print("Cannot plot due to optimization failure")
        return
    
    # Generate data for plots
    R_values, profit_values = generate_profit_curve(theta_opt)
    theta_values, wtp_values = generate_wtp_curve(0.01, 0.99, R_opt)
    
    # Calculate key values
    P_opt = P_theta(theta_opt, R_opt)
    
    # Create plots
    plt.figure(figsize=(24, 8))
    
    # Plot 1: Profit vs indemnity
    plt.subplot(1, 3, 1)
    plt.plot(R_values, profit_values)
    plt.axvline(x=R_opt, color='r', linestyle='--', label=f'Optimal R={R_opt:.2f}')
    plt.title('Profit vs Indemnity')
    plt.xlabel('R (indemnity)')
    plt.ylabel('Profit')
    plt.legend()

    # Plot 2: Beta distribution with threshold
    plt.subplot(1, 3, 2)
    x_values = np.linspace(0, 1, 1000)
    plt.plot(x_values, beta.pdf(x_values, a, b))
    plt.axvline(x=theta_opt, color='r', linestyle='--', label=f'θ*={theta_opt:.4f}')
    plt.fill_between(x_values, 0, beta.pdf(x_values, a, b), 
                    where=(x_values >= theta_opt), alpha=0.3)
    plt.title(f'Beta({a},{b}) distribution with threshold θ* = {theta_opt:.4f}')
    plt.xlabel('θ (risk probability)')
    plt.ylabel('Density')
    plt.legend()

    # Plot 3: WTP function
    plt.subplot(1, 3, 3)
    plt.plot(theta_values, wtp_values)
    plt.axhline(y=P_opt, color='g', linestyle='--', label=f'P*={P_opt:.4f}')
    plt.axvline(x=theta_opt, color='r', linestyle='--', label=f'θ*={theta_opt:.4f}')
    plt.title('Willingness-to-Pay Function')
    plt.xlabel('θ (risk probability)')
    plt.ylabel('P(θ, R*)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./plots/one_insurance_contract.png')

######################################## Main ########################################
def main():
    optimize_contract()
    plot_insurance_model()

if __name__ == "__main__":
    main()