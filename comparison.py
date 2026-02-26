import matplotlib.pyplot as plt

# Data from your report tables
buses = [1, 2, 3, 4, 5, 6, 7, 8, 9]
python_v = [1.0400, 1.0250, 1.0250, 1.0258, 0.9956, 1.0127, 1.0258, 1.0159, 1.0324]
psse_nr_v = [1.0400, 1.0250, 1.0250, 1.0258, 0.9956, 1.0127, 1.0258, 1.0159, 1.0324] # Example

# Plot 1: Voltage Magnitude Comparison
plt.figure(figsize=(10, 5))
plt.plot(buses, python_v, 'ro-', label='Python NR')
plt.plot(buses, psse_nr_v, 'bs--', label='PSS/E NR')
# Add lines for GS and FDLF here...
plt.xlabel('Bus Number')
plt.ylabel('Voltage Magnitude (p.u.)')
plt.title('Comparison of Voltage Magnitudes across Methods')
plt.legend()
plt.grid(True)
plt.savefig('voltage_comparison.png')

# Plot 2: Convergence Comparison (Iterations)
methods = ['Python NR', 'PSS/E NR', 'PSS/E GS', 'PSS/E FDLF']
iterations = [4, 4, 32, 12] # Replace with your actual recorded values
plt.figure(figsize=(8, 5))
plt.bar(methods, iterations, color=['red', 'blue', 'green', 'orange'])
plt.ylabel('Number of Iterations')
plt.title('Convergence Speed Comparison')
plt.show()