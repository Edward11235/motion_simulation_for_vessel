import numpy as np
import matplotlib.pyplot as plt

m11 = 12  # kg
m22 = 24  # kg
m33 = 1.5  # kg*m^2
d11 = 6  # kg*s^-1
d22 = 8  # kg*s^-1
d33 = 1.35  # kg*m^2*s^-1

def T(eta):
    # from ego to inertial frame
    return np.array([[np.cos(eta[2]), -np.sin(eta[2]), 0], 
                     [np.sin(eta[2]), np.cos(eta[2]), 0], 
                     [0, 0, 1]])

def T_inv(eta):
    # from inertial frame to ego frame
    theta = eta[2]
    return np.array([[np.cos(theta), np.sin(theta), 0],
                      [-np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])

def C(v):
    return np.array([[0, 0, -m22*v[1]], 
                     [0, 0, m11*v[0]], 
                     [m22*v[1], -m11*v[0], 0]])

def D(v):
    # For simplification, let's use an identity matrix. Adjust as needed based on v.
    return np.array([[d11, 0, 0], 
                     [0, d22, 0], 
                     [0, 0, d33]])

def ASV_dynamics(eta, v, tau, tau_env):
    M_inv = np.linalg.inv(np.array([[m11, 0, 0], 
                                   [0, m22, 0], 
                                   [0, 0, m33]]))
    eta_dot = np.dot(T(eta), v)
    tau_env_in_ego_frame = np.dot(T_inv(eta), tau_env)
    v_dot = np.dot(M_inv, tau + tau_env_in_ego_frame - np.dot(C(v) + D(v), v))
    return eta_dot, v_dot

# Simulation parameters
time_step = 0.01
num_steps = 10**4

# Initial conditions
eta = np.array([0.0, 0.0, 0.0])  # [x, y, psi]
v = np.array([0.0, 0.0, 0.0])  # [u, v, w]

# Placeholder control and environmental forces/torques
tau = np.array([1, 0, 0.1])
tau_env = np.array([0.01, 0, 0])  # Here, tau_env is in global frame

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
plt.axis('equal')  # Set equal scaling (i.e., make circles circular) by changing axis limits.
plt.show()