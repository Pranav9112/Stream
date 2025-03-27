import pandas as pd
from Exp import *
import streamlit as st

def process_file(file):
    # Read file into a DataFrame
  try:
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
        df = pd.read_excel(file)
    else:
        raise FileError
  except Exception as exp:
   st.error(exp, icon="🚨")
    
    # Ensure required columns exist
    required_columns = ['z-data', 't-data', 'linear_transmittance', 'sample_length', 'beam_waist', 'pulse_width', 'wavelength', 'energy', 'saturation_intensity', 'beta', 'gamma']
    try:
      for col in required_columns:
          if col not in df.columns:
              raise ColError
    except Exception as exp:
      st.warning(exp, icon="🚨")
    
    # Read constants from file
    z = df['z-data']
    t = df['t-data']
    lt = df['linear_transmittance'][0]
    sl = df['sample_length'][0]
    bw = df['beam_waist'][0]
    pw = df['pulse_width'][0]
    wl = df['wavelength'][0]
    en = df['energy'][0]
    si = df['saturation_intensity'][0]
    be = df['beta'][0]
    ga = df['gamma'][0]
        
    return [z, t, lt, sl, pw, wl, en, si, be, ga]
  
