"""
Temana es la biblioteca de funciones para el análisis temporal de datos y generación de señales artificiales.

"""

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d, make_interp_spline
import random

# Definición del dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x_train, y_train):
        self.x_train = x_train #serie deteriorada
        self.y_train = y_train #serie recuperada en super resolución

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

## Definición de la red Encoder basada en capas convolucionales.
class CNNAutoencoder(nn.Module):
    """
    Una red neuronal convolucional para autoencodificación de señales temporales.

    Args:
        n_input (int): Número de características de entrada (longitud del submuestreo, 1000).
        n_hidden (int): Número de unidades ocultas en cada capa convolucional.
        n_output (int): Número de características de salida (longitud de la señal completa, 5000).

    """
    def __init__(self, n_input, n_hidden, n_output):
        super(CNNAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_input, n_hidden, kernel_size=1),
            nn.BatchNorm1d(n_hidden),
            #nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=1),
            nn.Conv1d(n_hidden, n_hidden, kernel_size=1),
            nn.BatchNorm1d(n_hidden),
            #nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(n_hidden, n_hidden, kernel_size=3),
            nn.BatchNorm1d(n_hidden),
            #nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(n_hidden, n_output, kernel_size=1),
            nn.BatchNorm1d(n_output),
            #nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Propagación hacia adelante de la red neuronal.

        Args:
            x (tensor): Tensor de entrada con la señal submuestreada (longitud 1000).

        Returns:
            tensor: Tensor de salida con la señal reconstruida (longitud 5000).
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Función para generar señales aleatorias con baya y alta resolución.

def genera_signal(x,xm):
    """"
        Entrada: 
            Vector de datos x para el súper muestreo.
            Vector de datos xm para el submuestreo.
        Salida:
            Señal del súpermuestreo y del submuestreo.

    """
    choose = np.random.randint(0,2,1)
    A = (2*np.random.rand()-1)*np.random.randint(1,6,1) #Amplitud
    B = (2*np.random.rand()-1)*np.random.randint(1,10,1) #frecuencia
    C = (2*np.random.rand()-1)*np.random.randint(1,10,1) #traslación

    if choose == 0:
        # Generamos la señal basados en el seno
        y = A*np.sin(B*x) + C
        ym = A*np.sin(B*xm) + C
        
    else:
        #Generamos la señal basados en el coseno
        y = A*np.cos(B*x) + C
        ym = A*np.cos(B*xm) + C
    return([y, ym])




def genera_ruido(x,xm):
    """"
        Entrada: 
            Vector de datos x para el súper muestreo.
            Vector de datos xm para el submuestreo.
        Salida:
            Ruido del súpermuestreo y del submuestreo.

    """
    choose = np.random.randint(0,2,1)
    A = (2*np.random.rand()-1)*np.random.uniform(0.1,0.5,1) #Amplitud
    B = (2*np.random.rand()-1)*np.random.randint(100,200,1) #frecuencia

    if choose == 0:
        # Generamos la señal basados en el seno
        r = A*np.sin(B*x)
        rm = A*np.sin(B*xm)
        
    else:
        #Generamos la señal basados en el coseno
        r = A*np.cos(B*x) 
        rm = A*np.cos(B*xm) 
    return([r, rm])




def read_data(filename):
    """
    Lee los datos desde un archivo txt.

    Entrada:
        filename: Nombre del archivo txt.

    Respuestas:
        Un tensor de torch con los datos del archivo txt.
    """

    # with open(filename, "r") as f:
    #     data = f.read()
    # data = np.array([float(x) for x in data.split("\n")])
    data = np.loadtxt(filename)
    data = torch.tensor(data)
    if data.type != torch.float32:
        data = data.float()
        
    return data


import torch.nn as nn

class Autoencoder(nn.Module):
    """
    Clase que implementa una red neuronal autoencoder.

    Args:
        n_input: Número de neuronas en la capa de entrada.
        n_hidden: Número de neuronas en cada capa escondida.
        n_output: Número de neuronas en la capa de salida.
    """

    def __init__(self, n_input, n_hidden, n_output):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output)
        )

    def forward(self, x):
        """
        Realiza la propagación hacia adelante de la red neuronal.

        Args:
        x: Entrada de la red neuronal.

        Returns:
        Salida de la red neuronal.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    # Definición del dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]
    

class CNNAutoencoder(nn.Module):
    """
    Una red neuronal convolucional para autoencodificación de señales temporales.

    Args:
        n_input (int): Número de características de entrada (longitud del submuestreo, 1000).
        n_hidden (int): Número de unidades ocultas en cada capa convolucional.
        n_output (int): Número de características de salida (longitud de la señal completa, 5000).

    """
    def __init__(self, n_input, n_hidden, n_output):
        super(CNNAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_input, n_hidden, kernel_size=1),
            nn.BatchNorm1d(n_hidden),
            #nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=1),
            nn.Conv1d(n_hidden, n_hidden, kernel_size=1),
            nn.BatchNorm1d(n_hidden),
            #nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(n_hidden, n_hidden, kernel_size=3),
            nn.BatchNorm1d(n_hidden),
            #nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(n_hidden, n_output, kernel_size=1),
            nn.BatchNorm1d(n_output),
            #nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Propagación hacia adelante de la red neuronal.

        Args:
            x (tensor): Tensor de entrada con la señal submuestreada (longitud 1000).

        Returns:
            tensor: Tensor de salida con la señal reconstruida (longitud 5000).
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
# Función generadora de splines de tensión. Tomado del libro de Análisis Numérico de
# David Kinkaid

def mspline_t(x, y, tau):
    n = len(x) - 1
    
    # Step 1: Calculate hi, alpha, beta, gamma
    h = np.diff(x)
    alpha = 1/h - tau/np.sinh(tau * h)
    beta = tau * np.cosh(tau * h)/np.sinh(tau * h) - 1/h
    gamma = tau**2 * np.diff(y)/h

    # Step 2: Set up the tridiagonal system A * Z = Y
    A = np.zeros((n+1, n+1))
    Y = np.zeros(n+1)
    
    A[0, 0] = 1
    A[n, n] = 1
    for i in range(1, n):
        A[i, i-1] = alpha[i-1]
        A[i, i] = beta[i-1] + beta[i]
        A[i, i+1] = alpha[i]
        Y[i] = gamma[i] - gamma[i-1]
    
    # Solve the system for Z
    Z = np.linalg.solve(A, Y)
    
    # Step 3: Define the tension spline function
    def tension_spline(x_eval):
        result = np.zeros_like(x_eval)
        for i in range(n):
            mask = (x_eval >= x[i]) & (x_eval <= x[i+1])
            xi = x_eval[mask]
            t1 = Z[i] * np.sinh(tau * (x[i+1] - xi)) + Z[i+1] * np.sinh(tau * (xi - x[i]))
            t1 /= tau**2 * np.sinh(tau * h[i])
            t2 = (y[i] - Z[i]/tau**2) * (x[i+1] - xi) / h[i]
            t3 = (y[i+1] - Z[i+1]/tau**2) * (xi - x[i]) / h[i]
            result[mask] = t1 + t2 + t3
        return result
    
    return tension_spline

## Función para generar puntos de referencia de cambio de amplitud
def gen_puntos_ref(x0,x1):
    # Seleccionar aleatoriamente la amplitud (entre 1.0 y 5.0)
    choose = np.random.randint(0,2,1)
    A = (2*np.random.rand()-1)*np.random.randint(1,10,1) #Amplitud
    B = (2*np.random.rand()-1)*np.random.randint(1,10,1) #frecuencia
    C = (np.random.rand())*np.pi #fase      
        
    # Seleccionar aleatoriamente el número de puntos (entre 1 y 4)
    num_puntos = random.randint(1, 4)
    # print(num_puntos)
    
    # Generar puntos en el rango de 0 a 4 * pi
    partition = np.linspace(x0,x1,num_puntos+2)
    puntos = []
    
    for i in range(num_puntos+2):
        x = partition[i]
        y = (A * np.sin(B * (x-C)))[0]
        y = np.abs(y)
        # Nos aseguramos que la amplitud no sea baja
        if y < 0.5:
            y = (A * np.random.uniform(0.5,1))[0]
            y = np.abs(y)
        puntos.append((x, y))
    
    return puntos


def msplin_uno(puntos):
    # Separar las coordenadas x e y de los puntos
    x_coords, y_coords = zip(*puntos)
    
    # Crear la función de interpolación con un spline de grado uno (lineal)
    interpolador = interp1d(x_coords, y_coords, kind='linear', fill_value="extrapolate")
    
    # Crear una nueva serie de puntos para obtener la curva suavizada
    # x_nuevo = np.linspace(min(x_coords), max(x_coords), 100)
    # y_nuevo = interpolador(x_nuevo)
    
    return interpolador

def msplin_zero(puntos):
    # Separar las coordenadas x e y de los puntos
    x_coords, y_coords = zip(*puntos)
    
    # Crear la función de interpolación con un spline de grado uno (lineal)
    interpolador = interp1d(x_coords, y_coords, kind='zero', fill_value="extrapolate")
    
    # Crear una nueva serie de puntos para obtener la curva suavizada
    # x_nuevo = np.linspace(min(x_coords), max(x_coords), 100)
    # y_nuevo = interpolador(x_nuevo)
    
    return interpolador

def mbsplin_n(puntos,n):
    # Separar las coordenadas x e y de los puntos
    x_coords, y_coords = zip(*puntos)
    
    # Crear la función de interpolación con un spline de grado uno (lineal)
    interpolador = make_interp_spline(x_coords, y_coords, k=n)
    
    # Crear una nueva serie de puntos para obtener la curva suavizada
    # x_nuevo = np.linspace(min(x_coords), max(x_coords), 100)
    # y_nuevo = interpolador(x_nuevo)
    
    return interpolador

# Esta función genera señales de amplitud variable de acuerdo a los splines generados anteriormente

def genera_signalv(x,xm):
    """"
        Entrada: 
            Vector de datos x para el súper muestreo.
            Vector de datos xm para el submuestreo.
        Salida:
            Señal del súpermuestreo y del submuestreo.

    """
    choose = np.random.randint(0,2,1)
    choose2 = np.random.randint(0,2,1)
    A = (2*np.random.rand()-1)*np.random.randint(1,6,1) #Amplitud
    B = (2*np.random.rand()-1)*np.random.randint(1,25,1) #frecuencia
    C = (2*np.random.rand()-1)*np.random.randint(1,10,1) #traslación
    D = (np.random.rand())*np.pi #fase  
    puntos = gen_puntos_ref(xm[0],xm[-1])

    if choose == 0:
        # Generamos la señal basados en el seno
        y = A*np.sin(B*(x-D))
        ym = A*np.sin(B*(xm-D))         
    else:
        #Generamos la señal basados en el coseno
        y = A*np.cos(B*(x-D)) 
        ym = A*np.cos(B*(xm-D)) 
    
    puntos = gen_puntos_ref(x[0],x[-1])
    xs, ys = zip(*puntos) # Desempaquetamos los puntos
    tau = np.random.choice([1, 3, 5, 8, 10, 12, 15, 20])
    
    if choose2 == 0:        #Splines de tensión aleatoria
        y = mspline_t(xs,ys,tau)(x)*y + C
        ym = mspline_t(xs,ys,tau)(xm)*ym + C
    else:               #Splines de tensión ifinita
        y = msplin_zero(puntos)(x)*y + C
        ym = msplin_zero(puntos)(xm)*ym + C

    return([y, ym])

########################################################################
def genPuntosFreqNU_H(x0,x1):
        
    # Elegimos aleatoriamente frecuencias altas
    high_frecuency = np.random.uniform(20,100,10)
    high_frecuency = np.sort(high_frecuency)
    # Elegimos aleatoriamente frecuencias bajas
    low_frecuency = np.random.uniform(1,5,10)
    low_frecuency = np.sort(low_frecuency)
    
    # elegimos aleatoriamente el número de puntos de cambio de frecuencia
    num_puntos = random.randint(2, 11)    
    tdom = np.zeros(num_puntos + 2) # Para almacenar el 0 y el 1
    tdom[-1] = 1
    tdom[1:-1] = np.sort(np.random.rand(num_puntos)) 
    partition = x0 + (x1 - x0) * tdom 
    #partition = np.linspace(x0,x1,num_puntos+2)
    puntos = []
    tipo = []
    tdom_h = [tdom[0]] # Solo incluye información de altas frecunecias
    y_h = []
    
    # Inicializo el tipo de variación
    tipo_variacion = np.random.choice(["low","high"],p=[0.96,0.04])
    # Inicializo la frecuenca
    y = 1
    for i in range(num_puntos + 2):
        # Solo para la primera vez
        x = partition[i]
        
        if i == 0:            
            if tipo_variacion == "low":
                y = np.random.choice(low_frecuency)
                tipo.append("low")
                y_h.append(0)
            else:
                y = np.random.choice(low_frecuency) # Elige una frecuencia baja para transportar la frecuencia alta
                tipo.append("high")
                tdom_h.append(np.random.uniform(partition[i],partition[i+1],1)[0])
                yh_temporal = np.random.choice(high_frecuency)
                y_h.append(yh_temporal)
                y_h.append(yh_temporal)

        else:
            # forzamos a cambiar las frecuencias altas 
            if tipo[-1] == "high":
                tipo_variacion = np.random.choice(["low","no_change"],p=[0.95,0.05])
            else:
                tipo_variacion = np.random.choice(["low","high","no_change"],p=[0.20,0.07,0.73])
            if tipo_variacion == "low":
                y = np.random.choice(low_frecuency)
                y_h.append(0)
                tdom_h.append(x)
                tipo.append(tipo_variacion)

            elif tipo_variacion == "high":
                y = np.random.choice(low_frecuency) # Elige una frecuencia baja para transportar la frecuencia alta
                tipo.append("high")
                tdom_h.append(x)                
                if i != num_puntos+1:
                    yh_temporal = np.random.choice(high_frecuency)
                    tdom_h.append(np.random.uniform(partition[i],partition[i+1],1)[0])                    
                    y_h.append(yh_temporal)
                    y_h.append(yh_temporal)
                else:
                    y_h.append(0)              
                    
            else:
                tipo.append(tipo[-1])
                y_h.append(0)
                tdom_h.append(x)
            
        #print("y= ",y)
        puntos.append((x, y))
    puntos_h = [(tdom_h[i],y_h[i]) for i in range(len(tdom_h))]
    
    return (puntos,puntos_h,tipo)

########################################################################
########################################################################
def generaSignalFvNU_H(x,xm):
    """"
        Entrada: 
            Vector de datos x para el súper muestreo.
            Vector de datos xm para el submuestreo.
        Salida:
            Señal del súpermuestreo y del submuestreo.

    """
    # Generamos la función de cambio de frecuencias
    puntos, puntos_h, tipo = genPuntosFreqNU_H(x[0],x[-1])
    xf, yf = zip(*puntos) # Desempaquetamos los puntos
    tau = np.random.choice([1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2])
    # print("Tipo",tipo)
    # print("tau:",tau)
    
    # El mismo peso a las ondas con seno y coseno
    choose = np.random.choice([0, 1],1)
    # Se da menos peso a las variacione bruscas de frecuencia.
    choose2 = np.random.choice([0,1],1,p=[0.98,0.02])

    A = (2*np.random.rand()-1)*np.random.randint(1,6,1) #Amplitud
    B = mspline_t(xf,yf,tau) #frecuencia
    C = (2*np.random.rand()-1)*np.random.randint(1,10,1) #traslación
    D = (np.random.rand())*np.pi #fase  
    #puntos = tm.gen_puntos_ref(xm[0],xm[-1])


    if choose == 0:
        # Generamos la señal basados en el seno
        y = A*np.sin(B(x)*(x-D)) 
        ym = A*np.sin(B(xm)*(xm-D))    
    else:
        #Generamos la señal basados en el coseno
        y = A*np.cos(B(x)*(x-D)) 
        ym = A*np.cos(B(xm)*(xm-D)) 
    
    puntos = gen_puntos_ref(x[0],x[-1])
    xs, ys = zip(*puntos) # Desempaquetamos los puntos
    tau = np.random.choice(np.linspace(1,2,21))
    #print("tau2: ", tau)
    amplitud_ruido = np.random.uniform(0.08,0.2,1)
    
    ruido = amplitud_ruido*mspline_t(xs,ys,tau)(x)*np.sin(msplin_zero(puntos_h)(x)*x)
    ruidom = amplitud_ruido*mspline_t(xs,ys,tau)(xm)*np.sin(msplin_zero(puntos_h)(xm)*xm)
    
    if choose2 == 0:        #Splines de tensión aleatoria
        y = mspline_t(xs,ys,tau)(x)*y + C  + ruido
        ym = mspline_t(xs,ys,tau)(xm)*ym + C + ruidom
    else:               #Splines de tensión ifinita
        y = msplin_zero(puntos)(x)*y + C + ruido
        ym = msplin_zero(puntos)(xm)*ym + C + ruidom

    return([y, ym,ruido])
########################################################################
########################################################################

########################################################################
########################################################################
# Genera señales con diferentes tasas de muestreo., la idea es que luego se
# pueda comparar las reconstrucciones de los diferentes modelos con las 
# mismas señales. 
def generaListSignalFvNU_H(x,len_x):
    """"
        Entrada: 
            Vector de datos x para el súper muestreo.
            Lista de tamaños de señale del supermuestro  len_x = [250,500,1000]             
        Salida:
            Señal del súpermuestreo y de varias logitudes de submuestreo.
    """
    # Generamos la función de cambio de frecuencias
    puntos, puntos_h, tipo = genPuntosFreqNU_H(x[0],x[-1])
    list_x = [np.linspace(x[0],x[-1],lx) for lx in len_x] 
    xf, yf = zip(*puntos) # Desempaquetamos los puntos
    tau = np.random.choice([1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2])
    print("Tipo",tipo)
    print("tau:",tau)
    
    #choose = np.random.randint(0,2,1)
    choose = np.random.choice([0, 1],1)
    #choose2 = np.random.randint(0,2,1)
    choose2 = np.random.choice([0,1],1,p=[0.98,0.02])

    A = (2*np.random.rand()-1)*np.random.randint(1,6,1) #Amplitud
    B = mspline_t(xf,yf,tau) #frecuencia
    C = (2*np.random.rand()-1)*np.random.randint(1,10,1) #traslación
    D = (np.random.rand())*np.pi #fase  
    #puntos = tm.gen_puntos_ref(xm[0],xm[-1])


    if choose == 0:
        # Generamos la señal basados en el seno
        y = A*np.sin(B(x)*(x-D)) 
        list_y = [A*np.sin(B(xm)*(xm-D)) for xm in list_x]    
    else:
        #Generamos la señal basados en el coseno
        y = A*np.cos(B(x)*(x-D)) 
        list_y = [A*np.cos(B(xm)*(xm-D)) for xm in list_x] 
    
    puntos = gen_puntos_ref(x[0],x[-1])
    xs, ys = zip(*puntos) # Desempaquetamos los puntos
    tau = np.random.choice(np.linspace(1,2,21))
    print("tau2: ", tau)
    amplitud_ruido = np.random.uniform(0.08,0.2,1)
    
    ruido = amplitud_ruido*mspline_t(xs,ys,tau)(x)*np.sin(msplin_zero(puntos_h)(x)*x)
    list_ruido = [amplitud_ruido*mspline_t(xs,ys,tau)(xm)*np.sin(msplin_zero(puntos_h)(xm)*xm)
              for xm in list_x]
    
    if choose2 == 0:        #Splines de tensión aleatoria
        y = mspline_t(xs,ys,tau)(x)*y + C  + ruido
        list_y = [mspline_t(xs,ys,tau)(xm)*ym + C + ruidom for xm,ruidom,ym in zip(list_x,list_ruido,list_y)]
    else:               #Splines de tensión ifinita
        y = msplin_zero(puntos)(x)*y + C + ruido
        list_y = [msplin_zero(puntos)(xm)*ym + C + ruidom for xm,ruidom,ym in zip(list_x,list_ruido,list_y)]

    return([y, list_y,ruido])
########################################################################
########################################################################
