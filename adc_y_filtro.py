import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Funciones de procesamiento ---

def downsample(signal, fs_in, fs_out):
    """
    Reduce la frecuencia de muestreo de fs_in a fs_out por decimación simple.
    - signal: señal de entrada (1D numpy array)
    - fs_in: frecuencia de muestreo original
    - fs_out: frecuencia de muestreo deseada
    Retorna la señal decimada y la nueva fs (fs_out).
    """
    # Validar que fs_out < fs_in
    if fs_out >= fs_in:
        raise ValueError("fs_out debe ser menor que fs_in para downsampling.")
    decim_factor = fs_in / fs_out

    # Asegurarse de que el factor de decimación es entero para este modo simplificado
    if abs(decim_factor - round(decim_factor)) > 1e-9:
        raise ValueError("Solo se permite decimación entera en este modo simplificado.")
    
    M = int(round(decim_factor))
    return signal[::M], fs_out # Tomar cada M-ésimo punto


def quantize(signal, bits=8, v_min=None, v_max=None, scheme='mid-rise'):
    """
    Cuantiza la señal a un número de bits dado usando esquema mid-rise.
    - bits: número de bits de cuantización
    - v_min, v_max: rango de la señal
    - scheme: 'mid-rise' (estándar)
    Retorna la señal cuantizada, los índices de nivel y los valores de nivel.
    """

    L = 2 ** int(bits) # Niveles de cuantización
    v_min = float(np.min(signal)) if v_min is None else v_min
    v_max = float(np.max(signal)) if v_max is None else v_max

    if v_min == v_max:
        v_max = v_min + 1e-6 # Evitar división por cero
    
    s_clipped = np.clip(signal, v_min, v_max) # Recortar a rango
    delta = (v_max - v_min) / L # Paso de cuantización
    indices = np.floor((s_clipped - v_min) / delta).astype(int)
    indices = np.clip(indices, 0, L-1)
    levels_values = v_min + (indices + 0.5) * delta # Valores de nivel (mid-rise)

    return levels_values, indices, levels_values


def luenberger_filter(signal, L=0.05, a=1.0, c=1.0):
    """
    Filtro de Luenberger 1D para señal.
    - L: ganancia del observador
    - a, c: parámetros del sistema (asumidos 1D y conocidos)
    Retorna la señal filtrada.
    """
    x_hat = np.zeros_like(signal)

    # Ciclo para la predicción y actualización
    for k in range(1, len(signal)):
        # Actualización del estado estimado
        x_hat[k] = a * x_hat[k-1] + L * (signal[k-1] - c * x_hat[k-1])
    
    return x_hat


def kalman_filter(signal, Q=1e-5, R=1e-2):
    """
    Filtro de Kalman 1D para señal.
    - Q: varianza del proceso
    - R: varianza de la medición
    Retorna la señal filtrada.
    """
    n = len(signal)
    x_hat = np.zeros(n)
    P = np.zeros(n)
    x_hat[0] = signal[0]  # Inicializa con el primer valor
    P[0] = 1.0

    for k in range(1, n):
        # Predicción
        x_pred = x_hat[k-1]
        P_pred = P[k-1] + Q

        # Actualización
        K = P_pred / (P_pred + R)  # Ganancia de Kalman
        x_hat[k] = x_pred + K * (signal[k] - x_pred)
        P[k] = (1 - K) * P_pred

    return x_hat


# --- Parámetros de usuario ---

CSV_PATH = "A_DeviceMotion_data/wlk_7/sub_1.csv"
COLUMN = "userAcceleration.x"
FS_HIGH = 50.0  # Hz, frecuencia de muestreo original
FS_OUT = 25.0   # Hz, frecuencia de muestreo deseada
BITS = 8        # Bits de cuantización

# --- Carga y preprocesamiento de datos ---

def load_signal(csv_path, column):
    df = pd.read_csv(csv_path)
    return df[column].values


def main():
    # Cargar señal
    signal = load_signal(CSV_PATH, COLUMN)
    v_min, v_max = float(np.min(signal)), float(np.max(signal))

    # Filtrado
    signal_luenberger = luenberger_filter(signal, L=0.08) # Ajustar L segun la señal
    signal_kalman = kalman_filter(signal, Q=1e-3, R=2e-2) # Ajustar Q y R segun la señal

    # Downsampling
    signal_ds_luenberger, _ = downsample(signal_luenberger, FS_HIGH, FS_OUT)
    signal_ds_kalman, _ = downsample(signal_kalman, FS_HIGH, FS_OUT)

    # Cuantización
    signal_q_luenberger, _, _ = quantize(signal_ds_luenberger, bits=BITS, v_min=v_min, v_max=v_max)
    signal_q_kalman, _, _ = quantize(signal_ds_kalman, bits=BITS, v_min=v_min, v_max=v_max)

    # Ejes de tiempo
    t = np.arange(len(signal)) / FS_HIGH
    t_ds = np.arange(len(signal_ds_luenberger)) / FS_OUT

    # --- Gráficas ---
    plt.figure(figsize=(12,8))
    plt.subplot(3,1,1)
    plt.plot(t, signal, label=f'Original ({FS_HIGH} Hz)')
    plt.title(f'Señal original: {COLUMN} (fs = {FS_HIGH} Hz)')
    plt.ylabel('Amplitud')
    plt.xlim(0, 10)
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(t_ds, signal_ds_luenberger, '.-', label=f'Luenberger + Decimación ({FS_OUT} Hz)')
    plt.plot(t_ds, signal_ds_kalman, '.-', label=f'Kalman + Decimación ({FS_OUT} Hz)')
    plt.title('Señal filtrada y decimada')
    plt.ylabel('Amplitud')
    plt.xlim(0, 10)
    plt.legend()

    plt.subplot(3,1,3)
    plt.step(t_ds, signal_q_luenberger, where='mid', label=f'Luenberger + Cuantizada ({BITS} bits)')
    plt.step(t_ds, signal_q_kalman, where='mid', label=f'Kalman + Cuantizada ({BITS} bits)')
    plt.title(f'Señal cuantizada ({BITS} bits, mid-rise)')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.xlim(0, 10)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()