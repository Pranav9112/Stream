import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Custom exceptions
class FileFormatError(Exception):
    pass

class MissingColumnError(Exception):
    pass

# Solver Class
class ZScanSolution:
    def __init__(self, linear_transmittance, sample_length, beam_waist, pulse_width,
                 wavelength, z_values, t_values):
        self.L_T = linear_transmittance
        self.S_L = sample_length
        self.W_o = beam_waist
        self.P_W = pulse_width
        self.L_ = wavelength
        self.Z_d = z_values
        self.T_d = t_values

    @staticmethod
    def func(t, i, alfa0, i_s, beta, gamma):
        return -((alfa0 * i_s / (i_s + i)) + (beta * i) + (gamma * i ** 2)) * i

    def compute_transmittance(self, saturation_intensity, beta, gamma, energy, max_step):
        I_s = saturation_intensity
        zo = (np.pi * self.W_o ** 2) / self.L_
        a_0 = -np.log(self.L_T) / self.S_L
        I_oo = energy / ((np.pi * self.W_o ** 2) * self.P_W)

        transmittance = []
        for z in self.Z_d:
            I_initial = I_oo / (1 + (z / zo) ** 2)
            sol = solve_ivp(self.func, [0, self.S_L], [I_initial],
                            args=(a_0, I_s, beta, gamma),
                            method='RK45', max_step=max_step)
            I_output = sol.y[0, -1] if sol.success else I_initial
            transmittance.append(I_output / I_initial)

        new_trans = np.mean(transmittance[:10]) if transmittance else 1.0
        return np.array(transmittance) / new_trans

    @staticmethod
    def process_file(file):
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            raise FileFormatError("Unsupported file format")

        if 'z_values' not in df.columns or 't_values' not in df.columns:
            raise MissingColumnError("File must contain 'z_values' and 't_values' columns (case-sensitive)")

        return df

# Streamlit App
st.title("🔬 Z-Scan Simulation")
st.sidebar.header("Input Parameters")

# Sidebar inputs
linear_transmittance = st.sidebar.slider("Linear Transmittance", 0.1, 1.0, value=0.5)
sample_length = st.sidebar.number_input("Sample Length (m)", min_value=1e-100, format='%e', value=1e-3)
beam_waist = st.sidebar.number_input("Beam Waist at Focus (m)", min_value=1e-100, format='%e', value=1e-3)
pulse_width = st.sidebar.number_input("Pulse Width (s)", min_value=1e-100, format='%e', value=1e-9)
wavelength = st.sidebar.number_input("Wavelength (m)", min_value=1e-100, format='%e', value=1e-9)
energy = st.sidebar.number_input("Pulse Energy (J)", min_value=1e-100, format='%e', value=1e-3)
saturation_intensity = st.sidebar.number_input("Saturation Intensity (W/m²)", min_value=1e-100, format='%e', value=1e-6)
beta = st.sidebar.number_input("Beta (Nonlinear Absorption)", min_value=1e-100, format='%e', value=1e-6)
gamma = st.sidebar.number_input("Gamma (Nonlinear Refraction)", min_value=1e-100, format='%e', value=1e-6)
max_step = st.sidebar.number_input("Max Step Size for Solver", min_value=1e-6, value=1e-4, format='%e')

# File upload
v_file = st.file_uploader("📁 Upload Data File (.csv or .xlsx)")

# File and simulation processing
if v_file is not None:
    with st.expander("📘 File Format Instructions"):
        st.markdown("""
        - Your file must contain at least the following **numeric** columns:
        - `z_values` (in cm)
        - `t_values` (raw transmittance)
        - Columns must be **case-sensitive**.
        """)

    try:
        df = ZScanSolution.process_file(v_file)
        st.success("File loaded successfully!")
        st.write("### Preview of Uploaded Data", df.head())

        z_values = df['z_values'].values * 1e-2  # convert cm to m
        t_values = df['t_values'].values / 0.95  # normalize

        solver = ZScanSolution(linear_transmittance, sample_length, beam_waist,
                               pulse_width, wavelength, z_values, t_values)

        computed = solver.compute_transmittance(saturation_intensity, beta, gamma, energy, max_step)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(z_values, computed, 'r*', label="Simulated Data")
        ax.plot(z_values, t_values, 'b-o', label="Experimental Data")
        ax.set_xlabel("z (m)")
        ax.set_ylabel("Transmittance")
        ax.set_title("Z-Scan Comparison")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {str(e)}")

# Save parameters to file
params = [linear_transmittance, sample_length, beam_waist, pulse_width, wavelength,
          energy, saturation_intensity, beta, gamma, max_step]
with open('data.txt', 'a+') as f:
    f.write(",".join(map(str, params)) + "\n")
