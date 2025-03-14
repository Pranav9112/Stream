import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class ZScanSolution:
    def __init__(self, linear_transmittance, sample_length, beam_waist, pulse_width, wavelength, z_values, t_values):
        self.L_T = linear_transmittance  # Linear transmittance
        self.S_L = sample_length         # Sample length
        self.W_o = beam_waist            # Beam waist at focus
        self.P_W = pulse_width           # Pulse width
        self.L_ = wavelength             # Wavelength
        self.Z_d = z_values              # z values
        self.T_d = t_values              # Experimental transmittance values

    @staticmethod
    def func(t, i, alfa0, i_s, beta, gamma):
        return -((alfa0 * i_s / (i_s + i)) + (beta * i) + (gamma * i ** 2)) * i

    def compute_transmittance(self, saturation_intensity, beta, gamma, energy):
        I_s = saturation_intensity
        zo = (np.pi * self.W_o ** 2) / self.L_  # Rayleigh length
        a_0 = -np.log(self.L_T) / self.S_L      # Absorption coefficient
        I_oo = energy / ((np.pi * self.W_o ** 2) * self.P_W)  # Intensity at focus

        transmittance = []
        for z in self.Z_d:
            I_initial = I_oo / (1 + (z / zo) ** 2)  # Input intensity
            sol = solve_ivp(self.func, [0, self.S_L], [I_initial], args=(a_0, I_s, beta, gamma), method='RK45', max_step=max_step)
            if sol.success:
                I_output = sol.y[0, -1]
                transmittance.append(I_output / I_initial)
            else:
                transmittance.append(1.0)  # If solver fails, default to 1

        new_trans = np.mean(transmittance[:10])
        return np.array(transmittance) / new_trans

# Streamlit UI
st.title("Z-Scan Simulation")
st.sidebar.header("Input Parameters")

# User inputs
linear_transmittance = st.sidebar.slider("Linear Transmittance", 0.1, 1.0, 0.8)
sample_length = st.sidebar.number_input("Sample Length (m)", min_value=1e-100, format='%e')
beam_waist = st.sidebar.number_input("Beam Waist at Focus (m)", min_value=1e-100, format='%e')
pulse_width = st.sidebar.number_input("Pulse Width (s)", min_value=1e-100, format='%e')
wavelength = st.sidebar.number_input("Wavelength (m)", min_value=1e-100, format='%e')
energy = st.sidebar.number_input("Pulse Energy (J)", min_value=1e-100, format='%e')
saturation_intensity = st.sidebar.number_input("Saturation Intensity (W/m²)", min_value=1e-100, format='%e')
beta = st.sidebar.number_input("Beta (Nonlinear Absorption Coeff.)", min_value=1e-100, format='%e')
gamma = st.sidebar.number_input("Gamma (Nonlinear Refraction Coeff.)", min_value=1e-100, format='%e')
max_step = st.sidebar.number_input("Max Step Size for Solver", min_value=1e-6, value=1e-4, format='%e')

# User input for z and t data
st.sidebar.subheader("Z-Scan Data Inputs")
z_values_input = st.sidebar.text_area("Enter z values (comma-separated)")
t_values_input = st.sidebar.text_area("Enter t values (comma-separated)")
v_file = st.sidebar.file_uploader('Upload Data File', type = ['csv', 'excel'])

z_values = np.array([float(x) for x in z_values_input.split(',')]) * 1e-2
t_values = np.array([float(x) for x in t_values_input.split(',')]) / 0.95

# Run simulation
solver = ZScanSolution(linear_transmittance, sample_length, beam_waist, pulse_width, wavelength, z_values, t_values)
computed_transmittance = solver.compute_transmittance(saturation_intensity, beta, gamma, energy)

# Plot results
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(z_values, computed_transmittance, 'r*', label="Simulated Data")
ax.plot(z_values, t_values, 'b-o', label="Experimental Data")
ax.set_xlabel("z (m)")
ax.set_ylabel("Transmittance")
ax.set_title("Z-Scan Data")
ax.legend()
ax.grid()
st.pyplot(fig)

# 
