import glob
import os
import matplotlib.pyplot as plt
import numpy as np

# Path base dove si trovano i file
PATH = '/Users/gabri/OneDrive/Fisica/TESI/QPO_python/QPO_swift/Obs1/ALLDU_v/'

freq_files_list = os.path.join(PATH, 'global_freq_array_*.npy')
freq_files = sorted(glob.glob(freq_files_list))


plt.figure(figsize=(10, 6))


for freq_file in freq_files:
  # prendo ebin
  filename = os.path.basename(freq_file)
  ebin = filename.replace('global_freq_array_', '').replace('.npy', '')

  power_norm_file = os.path.join(PATH, f'global_norm_power_array_{ebin}.npy')

  freq = np.load(freq_file)
  power = np.load(power_norm_file)
  plt.plot(freq,power,marker='.',linestyle='')

  
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Normalized Power')
plt.title('Global Power Spectrum Comparison')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()

plt.show()