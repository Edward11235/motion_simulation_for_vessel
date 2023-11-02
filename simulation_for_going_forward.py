import numpy as np
import matplotlib.pyplot as plt

m11 = 12  # kg
m22 = 24  # kg
m33 = 1.5  # kg*m^2
d11 = 6  # kg*s^-1
d22 = 8  # kg*s^-1
d33 = 1.35  # kg*m^2*s^-1

# Placeholder functions for the matrices
def T(eta):
    # For simplification, let's use an identity matrix. Adjust as needed.
    return np.array([[np.cos(eta[2]), -np.sin(eta[2]), 0], 
                     [np.sin(eta[2]), np.cos(eta[2]), 0], 
                     [0, 0, 1]])

def C(v):
    # For simplification, let's use an identity matrix. Adjust as needed.
    return np.array([[0, 0, -m22*v[1]], 
                     [0, 0, m11*v[0]], 
                     [m22*v[1], -m11*v[0], 0]])

def D(v):
    # For simplification, let's use an identity matrix. Adjust as needed based on v.
    return np.array([[d11, 0, 0], 
                     [0, d22, 0], 
                     [0, 0, d33]])

def ASV_dynamics(eta, v, tau, tau_env):
    M_inv = np.linalg.inv(np.array([[12, 0, 0], 
                                   [0, 24, 0], 
                                   [0, 0, 1.5]]))  # Identity matrix as placeholder. Adjust as needed.
    eta_dot = np.dot(T(eta), v)
    v_dot = np.dot(M_inv, tau + tau_env - np.dot(C(v) + D(v), v))
    return eta_dot, v_dot

# Simulation parameters
time_step = 0.01
num_steps = 1000

# Initial conditions
eta = np.array([0.0, 0.0, 0.0])  # [x, y, psi]
v = np.array([0.0, 0.0, 0.0])  # [u, v, w]

# Placeholder control and environmental forces/torques
tau = np.array([1, 0, 0])
tau_env = np.array([0, 0, 0])

# Lists to store x and y trajectories
x_trajectory = [eta[0]]
y_trajectory = [eta[1]]

# Simulation
for _ in range(num_steps):
    eta_dot, v_dot = ASV_dynamics(eta, v, tau, tau_env)
    
    # Update states using simple Euler integration
    eta += eta_dot * time_step
    v += v_dot * time_step

    x_trajectory.append(eta[0])
    y_trajectory.append(eta[1])

# Plot
plt.plot(x_trajectory, y_trajectory)
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('Trajectory of the ASV')
plt.grid(True)
plt.show()