import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Create output directory
import os
os.makedirs('/home/vinish/Dropbox/Machine Learning/book/outputs', exist_ok=True)

# ============================================
# Slide 1: Mountain with hiker at top
# ============================================
fig, ax = plt.subplots(figsize=(12, 8))

# Create mountain profile (inverted parabola)
x = np.linspace(-5, 5, 1000)
y = -0.3 * x**2 + 5  # Mountain peak at x=0

# Fill mountain
ax.fill_between(x, y, -2, alpha=0.7, color='#8B7355', label='Mountain')
ax.plot(x, y, 'k-', linewidth=2)

# Add snow cap at peak
peak_x = x[np.abs(x) < 0.5]
peak_y = y[np.abs(x) < 0.5]
ax.fill_between(peak_x, peak_y, peak_y.min(), alpha=0.9, color='white')

# Hiker at top
ax.plot(0, 5, 'ro', markersize=20, label='You (at peak)')

# Arrows showing 4 directions
arrow_props = dict(arrowstyle='->', lw=3, color='blue')
directions = [
    (0, 5, -1.5, -0.5, 'West'),
    (0, 5, 1.5, -0.5, 'East'),
    (0, 5, -0.3, 1, 'North'),
    (0, 5, 0.3, 1, 'South')
]

for x_start, y_start, dx, dy, label in directions:
    arrow = FancyArrowPatch((x_start, y_start), (x_start + dx, y_start + dy),
                           **arrow_props)
    ax.add_patch(arrow)
    ax.text(x_start + dx*1.2, y_start + dy*1.2, label, fontsize=12, 
            ha='center', weight='bold')

# Steepest descent path
path_x = np.array([0, -0.5, -1.0, -1.5, -2.0, -2.5])
path_y = -0.3 * path_x**2 + 5
ax.plot(path_x, path_y, 'g--', linewidth=3, label='Steepest descent', alpha=0.7)

# Goal at bottom
ax.plot(-2.5, path_y[-1], 'g*', markersize=30, label='Goal (minimum)')

ax.set_xlim(-6, 6)
ax.set_ylim(-2, 7)
ax.set_xlabel('Position', fontsize=14, weight='bold')
ax.set_ylabel('Elevation', fontsize=14, weight='bold')
ax.set_title('The Mountain Analogy: Finding the Steepest Descent', 
             fontsize=16, weight='bold', pad=20)
ax.legend(loc='upper right', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/vinish/Dropbox/Machine Learning/book/outputs/slide1_mountain_analogy.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# Slide 2: Gradient Direction Visualization
# ============================================
fig, ax = plt.subplots(figsize=(12, 8))

# Create loss surface (2D parabola)
x = np.linspace(-3, 3, 100)
y = x**2 + 2  # Simple parabola

# Color regions
uphill_x = x[x > 0]
uphill_y = uphill_x**2 + 2
downhill_x = x[x < 0]
downhill_y = downhill_x**2 + 2

ax.fill_between(uphill_x, uphill_y, 0, alpha=0.3, color='red', label='Uphill (gradient points here)')
ax.fill_between(downhill_x, downhill_y, 0, alpha=0.3, color='green', label='Downhill (we move here)')
ax.plot(x, y, 'b-', linewidth=3, label='Loss function (MSE)')

# Mark minimum
ax.plot(0, 2, 'o', color='gold', markersize=25, label='Minimum', markeredgecolor='black', markeredgewidth=2)

# Current position
current_pos = 2.5
current_loss = current_pos**2 + 2
ax.plot(current_pos, current_loss, 'ko', markersize=15, label='Current position')

# Gradient arrow (pointing uphill - right)
gradient_arrow = FancyArrowPatch((current_pos, current_loss), 
                                (current_pos + 0.8, current_loss + 3),
                                arrowstyle='->', lw=4, color='red', mutation_scale=30)
ax.add_patch(gradient_arrow)
ax.text(current_pos + 1.2, current_loss + 3.5, 'Gradient\n(uphill)', 
        fontsize=14, weight='bold', color='red', ha='center')

# Movement arrow (pointing downhill - left)
move_arrow = FancyArrowPatch((current_pos, current_loss), 
                            (current_pos - 0.8, current_loss - 1.5),
                            arrowstyle='->', lw=4, color='green', mutation_scale=30)
ax.add_patch(move_arrow)
ax.text(current_pos - 1.2, current_loss - 2, 'Our move\n(downhill)', 
        fontsize=14, weight='bold', color='green', ha='center')

ax.set_xlim(-3.5, 3.5)
ax.set_ylim(0, 15)
ax.set_xlabel('Parameter β', fontsize=14, weight='bold')
ax.set_ylabel('MSE (Loss)', fontsize=14, weight='bold')
ax.set_title('Gradient Points Uphill - We Move Downhill', 
             fontsize=16, weight='bold', pad=20)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/vinish/Dropbox/Machine Learning/book/outputs/slide2_gradient_direction.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# Slide 3-7: Gradient Descent Steps Animation
# ============================================
# Create 5 frames showing the stepping process
x = np.linspace(-3, 3, 100)
y = x**2 + 2

# Starting positions for descent
positions = [2.5, 1.8, 1.2, 0.6, 0.1]
colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']

for frame in range(5):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot loss function
    ax.plot(x, y, 'b-', linewidth=3, alpha=0.7, label='Loss function')
    
    # Mark minimum
    ax.plot(0, 2, 'o', color='gold', markersize=30, label='Target minimum', 
            markeredgecolor='black', markeredgewidth=3, zorder=10)
    
    # Show all previous steps
    for i in range(frame + 1):
        pos = positions[i]
        loss = pos**2 + 2
        
        # Plot position
        ax.plot(pos, loss, 'o', color=colors[i], markersize=20, 
                markeredgecolor='black', markeredgewidth=2, zorder=5)
        ax.text(pos, loss + 1.5, f'Step {i+1}', fontsize=12, 
                ha='center', weight='bold')
        
        # Draw arrow to next position (if not last)
        if i < frame and i < len(positions) - 1:
            next_pos = positions[i + 1]
            next_loss = next_pos**2 + 2
            arrow = FancyArrowPatch((pos, loss), (next_pos, next_loss),
                                   arrowstyle='->', lw=3, color='darkgreen',
                                   mutation_scale=25, zorder=3)
            ax.add_patch(arrow)
    
    # Highlight current position
    current_pos = positions[frame]
    current_loss = current_pos**2 + 2
    circle = Circle((current_pos, current_loss), 0.3, color='yellow', 
                   alpha=0.5, zorder=4)
    ax.add_patch(circle)
    
    # Show gradient at current position
    gradient_dx = 0.5
    gradient_dy = 2 * current_pos * gradient_dx  # derivative of x^2
    gradient_arrow = FancyArrowPatch((current_pos, current_loss), 
                                    (current_pos + gradient_dx, current_loss + gradient_dy),
                                    arrowstyle='->', lw=3, color='red',
                                    mutation_scale=20, linestyle='--', alpha=0.7)
    ax.add_patch(gradient_arrow)
    ax.text(current_pos + gradient_dx + 0.3, current_loss + gradient_dy, 
            'Gradient', fontsize=11, color='red', weight='bold')
    
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(0, 15)
    ax.set_xlabel('Parameter β', fontsize=14, weight='bold')
    ax.set_ylabel('MSE (Loss)', fontsize=14, weight='bold')
    ax.set_title(f'Gradient Descent: Step {frame + 1} of 5', 
                 fontsize=16, weight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add text box with info
    #textstr = f'Position: β = {current_pos:.2f}\nLoss: MSE = {current_loss:.2f}\nGradient: {2*current_pos:.2f}'
    #props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    #ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
    #        verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(f'/home/vinish/Dropbox/Machine Learning/book/outputs/slide{frame+3}_step_{frame+1}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

# ============================================
# Bonus: Learning Rate Comparison
# ============================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

x = np.linspace(-3, 3, 100)
y = x**2 + 2

learning_rates = [0.1, 0.5, 1.2]
titles = ['Too Small (η=0.1)', 'Just Right (η=0.5)', 'Too Large (η=1.2)']
descriptions = ['Slow convergence', 'Efficient descent', 'Overshooting!']

for idx, (eta, title, desc) in enumerate(zip(learning_rates, titles, descriptions)):
    ax = axes[idx]
    
    # Plot loss function
    ax.plot(x, y, 'b-', linewidth=2, alpha=0.7)
    ax.plot(0, 2, 'o', color='gold', markersize=20, markeredgecolor='black', markeredgewidth=2)
    
    # Simulate gradient descent
    beta = 2.5
    trajectory = [beta]
    
    for _ in range(8):
        gradient = 2 * beta
        beta = beta - eta * gradient
        trajectory.append(beta)
        if abs(beta) > 5:  # Stop if diverging
            break
    
    # Plot trajectory
    for i in range(len(trajectory) - 1):
        pos1, pos2 = trajectory[i], trajectory[i+1]
        loss1, loss2 = pos1**2 + 2, pos2**2 + 2
        
        if abs(pos2) > 4:  # Highlight overshooting
            color = 'red'
            alpha = 0.8
        else:
            color = 'green'
            alpha = 0.6
        
        ax.plot([pos1, pos2], [loss1, loss2], 'o-', color=color, 
                markersize=8, linewidth=2, alpha=alpha)
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 15)
    ax.set_xlabel('Parameter β', fontsize=12, weight='bold')
    ax.set_ylabel('MSE', fontsize=12, weight='bold')
    ax.set_title(f'{title}\n{desc}', fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3)

plt.suptitle('Impact of Learning Rate on Convergence', fontsize=16, weight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/slide8_learning_rate_comparison.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("All visualizations created successfully!")
print("\nGenerated files:")
print("1. slide1_mountain_analogy.png - Mountain hiking analogy")
print("2. slide2_gradient_direction.png - Gradient direction with color coding")
print("3-7. slide3-7_step_X.png - 5 frames showing stepping process")
print("8. slide8_learning_rate_comparison.png - Learning rate effects")