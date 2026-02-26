import matplotlib.pyplot as plt
import numpy as np

# Create a comprehensive dashboard figure
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Task 2: PSS/E Verification and Comparison - IEEE 9-Bus System', 
             fontsize=18, fontweight='bold')

# Grid layout
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Data
buses = [1, 2, 3, 4, 5, 6, 7, 8, 9]
python_vmag = [1.0400, 1.0250, 1.0250, 1.0258, 0.9956, 1.0127, 1.0258, 1.0159, 1.0324]
python_angle = [0.0000, 9.2800, 4.6648, -2.2168, -3.9888, -3.6874, 3.7197, 0.7275, 1.9667]

# Plot 1: Voltage Profile (top left)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(buses, python_vmag, 'bo-', linewidth=2, markersize=8, label='Python NR')
ax1.set_xlabel('Bus', fontsize=11)
ax1.set_ylabel('|V| (p.u.)', fontsize=11)
ax1.set_title('Voltage Magnitude Profile', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(buses)
ax1.set_ylim([0.99, 1.05])
ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

# Plot 2: Angle Profile (top middle)
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(buses, python_angle, 'rs-', linewidth=2, markersize=8, label='Python NR')
ax2.set_xlabel('Bus', fontsize=11)
ax2.set_ylabel('δ (degrees)', fontsize=11)
ax2.set_title('Voltage Angle Profile', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(buses)

# Plot 3: Method Comparison Table (top right)
ax3 = fig.add_subplot(gs[0, 2])
ax3.axis('tight')
ax3.axis('off')
table_data = [
    ['Method', 'Iterations', 'Mismatch (p.u.)', 'Converged?'],
    ['Python NR', '4', '3.4e-7', 'Yes'],
    ['PSS/E NR', '3', '0.00', 'Yes'],
    ['PSS/E GS', '18', '0.85', 'Yes'],
    ['PSS/E FDLF', '3', '0.04', 'Yes']
]
table = ax3.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.2, 0.15, 0.2, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)
for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold')
    elif i % 2 == 1:
        cell.set_facecolor('#D9E1F2')
ax3.set_title('Convergence Summary', fontsize=13, fontweight='bold', pad=20)

# Plot 4: Convergence Profile (middle left)
ax4 = fig.add_subplot(gs[1, :2])
python_iter = [1, 2, 3, 4]
python_mismatch = [1.63, 0.1875, 0.00215, 3.4e-7]
psse_nr_iter = [0, 1, 2, 3]
psse_nr_mismatch = [1.63, 0.0986, 0.00207, 7e-5]
gs_iter = list(range(1, 19))
gs_mismatch = [1632, 2883, 623, 371, 271, 191, 122, 92.3, 59.5, 34.4, 23.5, 13.9, 9.77, 6.40, 3.54, 2.22, 1.21, 0.87]

ax4.semilogy(python_iter, python_mismatch, 'bo-', linewidth=2, markersize=8, label='Python NR')
ax4.semilogy(psse_nr_iter, psse_nr_mismatch, 'rs-', linewidth=2, markersize=8, label='PSS/E NR')
ax4.semilogy(gs_iter, gs_mismatch, 'g*-', linewidth=1.5, markersize=4, label='PSS/E GS', alpha=0.7)
ax4.axhline(y=1e-4, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Tolerance')
ax4.set_xlabel('Iteration', fontsize=11)
ax4.set_ylabel('Max Mismatch (p.u.)', fontsize=11)
ax4.set_title('Convergence Profile Comparison', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3, which='both')
ax4.legend(loc='upper right', fontsize=10)
ax4.set_xlim([0, 18])
ax4.set_ylim([1e-7, 1e4])

# Plot 5: Line Flow Summary (bottom)
ax5 = fig.add_subplot(gs[2, :])
branches = ['1-4', '2-7', '3-9', '4-5', '4-6', '5-7', '6-9', '7-8', '8-9']
losses = [0.000, 0.000, 0.000, 0.258, 0.166, 2.300, 1.354, 0.475, 0.088]  # From Table 4
colors = ['gray' if l == 0 else 'orange' if l < 0.5 else 'red' for l in losses]
bars = ax5.bar(branches, losses, color=colors, alpha=0.7)
ax5.set_xlabel('Branch (From-To)', fontsize=12)
ax5.set_ylabel('Loss (MW)', fontsize=12)
ax5.set_title('Branch Power Losses (Total = 4.641 MW)', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')
for bar, loss in zip(bars, losses):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
             f'{loss:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('task2_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()