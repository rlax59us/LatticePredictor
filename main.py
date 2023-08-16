import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt

def visualize_waveform(x, sr, type):
    time = np.linspace(0, len(x)/sr, len(x))
    plt.cla()
    plt.plot(time, x)
    plt.ylabel('Amplitude')
    plt.xlabel('Time(s)')
    plt.savefig('./img/' + type+'.png', bbox_inches = 'tight')

def Get_White_Noise(length):
    rng = np.random.default_rng(seed=42)
    noise = rng.random(length)

    return noise

def AutoCorrelation(u, lpc_order=10):
    n = u.shape[-1]
    r = np.zeros((lpc_order+1))
    
    for m in range(lpc_order+1):
        for l in range(n):
            if l - m < 0:
                r[m] += 0 #since outside of the data is zero
            else:
                r[m] += u[l]*u[l-m]
        r[m] /= n

    return r

def Levinson_Durbin(r, lpc_order=10):
    n = r.shape[-1]
    a = np.zeros((lpc_order+1, n))
    kappa = np.zeros(lpc_order+1)
    delta = np.zeros(lpc_order+1)
    p = np.zeros(lpc_order+1)
    a[:, 0] = 1.0 #Since a_(m-1, 0) = 1
    p[0] = np.copy(r[0]) #Since P_0 = r(0)

    for m in range(1, lpc_order+1):
        #range(0, m-1)
        for l in range(m):
            delta[m-1] += r[l-m]*a[m-1, l]
        
        kappa[m] = delta[m-1] / np.maximum(1e-6, p[m-1])

        #a_(m, l) =  a_(m-1, l) + k_m a*_(m-1, m-l) for range(1, m)
        for l in range(1, m+1):
            a[m, l] = a[m-1, l] + kappa[m]*(-a[m-1, m-l])

        p[m] = p[m-1] * (1 - kappa[m]**2) #P_m = P_m-1(1-kamma**2)

    return kappa

#Using AutoCorrelation and Levinson-Durbin
def Lacttice_Predictor_A(lpc_order, u, kappa=[]):
    n = u.shape[-1]
    f = np.zeros((lpc_order+1, n))
    b = np.zeros((lpc_order+1, n))
    f[0] = u
    b[0] = u

    for m in range(1, lpc_order+1):
        for l in range(n):
            if l > 1:
                f[m][l] = f[m-1][l] - kappa[m]*b[m-1][l-1]
                b[m][l] = b[m-1][l-1] + kappa[m]*f[m-1][l]
            else:
                f[m][l] = f[m-1][l] 
                b[m][l] = kappa[m]*f[m-1][l]
    
    return f, b, kappa

#Using Partial Correlation
def Lacttice_Predictor_B(lpc_order, u):
    n = u.shape[-1]
    f = np.zeros((lpc_order+1, n))
    b = np.zeros((lpc_order+1, n))
    f[0] = u
    b[0] = u

    kappa = np.zeros(lpc_order+1)

    for m in range(1, lpc_order+1):
        numerator = 0
        denominator_left = 0
        denominator_right = 0

        for l in range(n):
            if l > 1:
                numerator += b[m-1][l-1]*(-f[m-1][l])
                denominator_left += b[m-1][l-1]**2
                denominator_right += f[m-1][l]**2
            else:
                continue #since outside of the data is zero
            
        numerator /= n
        denominator_left /= n
        denominator_right /= n
            
        #kappa is negative of partial correlation
        kappa[m] = -(numerator/np.sqrt(denominator_left*denominator_right)) 

        for l in range(n):
            if l > 1:
                f[m][l] = f[m-1][l] - kappa[m]*b[m-1][l-1]
                b[m][l] = b[m-1][l-1] + kappa[m]*f[m-1][l]
            else:
                f[m][l] = f[m-1][l] 
                b[m][l] = kappa[m]*f[m-1][l]
    
    return f, b, kappa

#f10 to f0
def All_Pole_Lattice_Filter(lpc_order, kappa, f_m, b):
    n = f_m.shape[-1]
    f = np.zeros((lpc_order+1, n))
    f[lpc_order] = f_m
    for m in range(lpc_order):
        for l in range(n):
            if l > 1:
                f[lpc_order-m-1][l] = f[lpc_order-m][l] + kappa[lpc_order-m]*b[lpc_order-m-1][l-1]
            else:
                f[lpc_order-m-1][l] = f[lpc_order-m][l] + 0

    return f[0]

if __name__ == '__main__':
    LPC_order = 10
    method = 'Levinson' #Partial or Levinson
    sr, u = wavfile.read('./wav/speech1.wav')
    visualize_waveform(x=u, sr=sr, type='original')
    n = u.shape[-1]
    white_noise = Get_White_Noise(length=n)

    if method == 'Partial':
        f, b, kappa = Lacttice_Predictor_B(lpc_order=LPC_order, u=u)
    else:
        r = AutoCorrelation(u=u)
        kappa = Levinson_Durbin(r=r,lpc_order=10)
        f, b, kappa = Lacttice_Predictor_A(lpc_order=LPC_order, u=u, kappa=kappa)

    #Give f10
    f0_from_f10 = All_Pole_Lattice_Filter(lpc_order=LPC_order, kappa=kappa, f_m=f[10], b=b)
    f0_from_f10 = np.asarray(f0_from_f10, dtype=np.int16)
    visualize_waveform(x=f0_from_f10, sr=sr, type=method+'_from_f10')
    wavfile.write('./wav/' + method+'_from_f10.wav', sr, f0_from_f10)
    #Give white noise
    f0_from_noise = All_Pole_Lattice_Filter(lpc_order=LPC_order, kappa=kappa, f_m=white_noise, b=b)
    f0_from_noise = np.asarray(f0_from_noise, dtype=np.int16)
    visualize_waveform(x=f0_from_noise, sr=sr, type=method+'_from_noise')
    wavfile.write('./wav/' + method+'_from_noise.wav', sr, f0_from_noise)




    