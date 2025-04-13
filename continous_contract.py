from config import lambda_, L, a, b
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import beta


def lagrangian_value(theta):
    """    Calculate the Lagrangian value for a given theta (array)."""
    return (1/lambda_) * np.log(theta * np.exp(lambda_*L) + 1 - theta) - L * theta

def profit():
    """    Calculate the profit of the contract. Returns a float."""
    return quad(lambda x: lagrangian_value(x)*beta.pdf(x, a, b), 0, 1)[0]

def plot_R():
    theta = np.linspace(0, 1, 100)
    lagrangian = lagrangian_value(theta)
    
    plt.plot(theta, lagrangian, label='L(theta)')
    plt.title('Lagrangian Function')
    plt.xlabel('Theta')
    plt.ylabel('Lagrangian')
    plt.legend()
    plt.grid()
    plt.savefig('plots/lagrangian_function.png')

def main():
    print(f'Profit: {profit():.4f}')
    plot_R()

if __name__ == "__main__":
    main() 
