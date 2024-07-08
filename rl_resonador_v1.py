# A Roger Arnau
# 2024 juny

# Basat en: ejemplo_optimizar_funcion_MSEL

import  numpy as np
import  random

import  matplotlib.pyplot as plt
from    keras               import Sequential, models
from    collections         import deque
from    keras.layers        import Dense
from    keras.optimizers    import Adam
from    keras.utils         import set_random_seed
from    math                import pi
from    tqdm                import tqdm
from    scipy.io            import savemat

# Hiperparameters (by the moment, import all)

from configs.dimensions_1 import *
from configs.config_rl_1  import *


# Functions

C = 299792458 
def sn(rn):
    return pi * rn**2
def vc(rc, lc):
    return pi * rc**2 * lc
def freq(rn, ln, rc, lc):
    return C / (2*pi) * ( sn(rn) / ( ln * vc(rc, lc) ) )**(1/2)

# Entorno

class Resonador():
    
    def __init__(self):
        self.recompensa = 0
        
    def Paso(self, action):
        
        f_previo    = self.f
        vc_previo   = self.vc
        self.recompensa = 0
        
        if action != 0:
            self.recompensa -= PENALIZACION_MOVIMIENTO
            if action == 1: self.rn += MOVIMIENTO
            if action == 2: self.rn -= MOVIMIENTO
            if action == 3: self.ln += MOVIMIENTO
            if action == 4: self.ln -= MOVIMIENTO
            if action == 5: self.rc += MOVIMIENTO
            if action == 6: self.rc -= MOVIMIENTO
            if action == 7: self.lc += MOVIMIENTO
            if action == 8: self.lc -= MOVIMIENTO
                
            self.rn = np.clip(self.rn, RANGO_RN[0], RANGO_RN[1])
            self.ln = np.clip(self.ln, RANGO_LN[0], RANGO_LN[1])
            self.rc = np.clip(self.rc, RANGO_RC[0], RANGO_RC[1])
            self.lc = np.clip(self.lc, RANGO_LC[0], RANGO_LC[1])
        
        self.Calcula_F()
        
        ### RECOMPENSAS
        
        if abs( self.f - f_previo ) / f_previo < CASI_0:
            if abs( self.vc - vc_previo ) / vc_previo < CASI_0:    
                self.recompensa += 0
            elif self.vc < vc_previo:
                self.recompensa += 2
            else :
                self.recompensa += -3
        elif self.f < f_previo:
            if abs( self.vc - vc_previo ) / vc_previo < CASI_0:    
                self.recompensa += 4
            elif self.vc < vc_previo:
                self.recompensa += 5
            else :
                self.recompensa += 0
        else :
            if abs( self.vc - vc_previo ) / vc_previo < CASI_0:    
                self.recompensa += -5
            elif self.vc < vc_previo:
                self.recompensa += -1
            else :
                self.recompensa += -6
        
        estado = [self.rn, self.ln, self.rc, self.lc]
        return estado, self.recompensa
    
    def Calcula_F(self):
        self.sn = sn(self.rn)
        self.vc = vc(self.rc, self.lc)
        self.f  = freq(self.rn, self.ln, self.rc, self.lc)
    
    def Reinicia(self):
        self.rn = INICIO_RN
        self.ln = INICIO_LN
        self.rc = INICIO_RC
        self.lc = INICIO_LC
        self.Calcula_F()
        return [self.rn, self.ln, self.rc, self.lc]
    
    def Reinicia_Aleatorio(self):
        # Número aleatorio entre los que se puede llegar empezando del inicio
        self.rn = random.choice([ RANGO_RN[0] + i*MOVIMIENTO for i in range(int((RANGO_RN[1]-RANGO_RN[0])/MOVIMIENTO + 1)) ])
        self.ln = random.choice([ RANGO_LN[0] + i*MOVIMIENTO for i in range(int((RANGO_LN[1]-RANGO_LN[0])/MOVIMIENTO + 1)) ])
        self.rc = random.choice([ RANGO_RC[0] + i*MOVIMIENTO for i in range(int((RANGO_RC[1]-RANGO_RC[0])/MOVIMIENTO + 1)) ])
        self.lc = random.choice([ RANGO_LC[0] + i*MOVIMIENTO for i in range(int((RANGO_LC[1]-RANGO_LC[0])/MOVIMIENTO + 1)) ])
        self.Calcula_F()
        return [self.rn, self.ln, self.rc, self.lc]


# Agente

class Agente():
    
    def __init__(self, rn_capas):
        self.paso       = 0
        self.epsilon    = EPSILON_INICIAL
        self.memoria    = deque(maxlen = int(1e6))
        self.modelo     = self.Crear_Modelo(rn_capas)
        
    def Guardar_Modelo(self, archivo):
        self.modelo.save(archivo)
    
    def Cargar_Modelo(self, archivo):
        self.modelo = models.load_model(archivo)
        
    def Crear_Modelo(self, rn_capas):
        modelo = Sequential()
        modelo.add( Dense(rn_capas[0],
                          input_shape = (ESPACIO_ESTADOS,),
                          activation = FUN_ACT) )
        for i in range(1,len(rn_capas)):
            modelo.add( Dense(rn_capas[i],
                              activation = FUN_ACT) )
        modelo.add( Dense(ESPACIO_ACCIONES,
                          activation='linear') )
        modelo.compile( loss = 'mse',
                        optimizer = Adam(learning_rate = TASA_APRENDIZAJE) )
        modelo.summary()
        return modelo
    
    def Guarda(self, estado, accion, recompensa, estado_siguiente):
        self.memoria.append( (estado, accion, recompensa, estado_siguiente) )
    
    def Actua(self, estado, deterministico = False) :
        
        self.paso += 1
        [Accion_recompensa] = self.modelo.predict([estado], verbose = 0)
        
        # Caso deterministico: siempre lo mejor
        if deterministico:
            return np.argmax( Accion_recompensa )
        
        # A veces, aleatorio
        if np.random.rand() <= self.epsilon:
            return random.choice( range(ESPACIO_ACCIONES) )
        
        # Elige el mejor con mayor probabilidad
        return np.argmax( Accion_recompensa )
    
    def Actualiza_Episodio(self) :
        if self.epsilon > EPSILON_MINIMO :
            self.epsilon *= EPSILON_DECREMENTO
    
    def Entrena(self):
        
        # Entrenamos con toda la memoria, pero se podria hacer sobre un subconjunto
        
        batch = self.memoria
        estados             = np.array([i[0] for i in batch])
        acciones            = np.array([i[1] for i in batch])
        recompensas         = np.array([i[2] for i in batch])
        estados_siguientes  = np.array([i[3] for i in batch])
        
        Q_actual = self.modelo.predict(estados, verbose = 0)
        Q_futura = self.modelo.predict(estados_siguientes, verbose = 0)
        Q_nueva  = recompensas + GAMMA * np.amax(Q_futura, axis = 1)
        
        indices = np.arange(len(batch))
        Q_actual[indices, acciones] = Q_nueva
        
        self.modelo.fit(x = estados, y = Q_actual, epochs = 1, verbose = 0)


def Muestra_Resultado(rn_total, ln_total, rc_total,
                      lc_total, vc_total, f_total,
                      recompensa_total, episodio) :
            
    fig, axs = plt.subplots(3, 1,
                            figsize=(10, 10),
                            height_ratios=[3,3,1])
    if episodio == None: episodio = 'determinístico'
    fig.suptitle(f'Evolución a lo largo del episodio {episodio}', fontsize=16)

    x = range(0, PASOS_POR_EPISODIO + 1)
    
    axs[0].plot(x, rn_total, label = 'Radio cuello')
    axs[0].plot(x, ln_total, label = 'Longitud Cuello')
    axs[0].plot(x, rc_total, label = 'Radio cavidad')
    axs[0].plot(x, lc_total, label = 'Longitud cavidad')
    axs[0].legend()

    axs[1].plot(x, vc_total, label = 'Volumen cavidad')
    axs1b = axs[1].twinx()
    axs1b.plot(x, f_total, 'k', label = 'Frecuencia')
    axs[1].legend(loc = 'upper left')
    axs1b.legend(loc = 'upper right')

    bar_colors = ['g' if val > 0 else 'r' for val in recompensa_total]
    axs[2].bar(x, recompensa_total, color=bar_colors)
    axs[2].axhline(y=0, color='g')
    axs[2].set_title('Recompensas')

    plt.tight_layout()
    plt.show()
    
    savemat(f'resultados/epoch_{episodio}.mat',
            {'rn_total': rn_total,
             'ln_total': ln_total,
             'rc_total': rc_total,
             'lc_total': lc_total,
             'vc_total': vc_total,
             'f_total' : f_total,
             'recompensa_total': recompensa_total,})
    
    
def Entrena_Agente(episodios = 1) :
    
    for e in range(episodios) :
        
        estado = entorno.Reinicia()
        if random.random() < PROB_INICIO_ALEATORIO :
            estado = entorno.Reinicia_Aleatorio()
            
        rn_total = [entorno.rn]
        ln_total = [entorno.ln]
        rc_total = [entorno.rc]
        lc_total = [entorno.lc]
        vc_total = [entorno.vc]
        f_total  = [entorno.f]
        recompensa_total = [entorno.recompensa]
        
        for p in tqdm(range(PASOS_POR_EPISODIO),
                      desc = f'Entrenando episodio {e+1} de {episodios}') :
            accion = agente.Actua(estado)
            estado_siguiente, recompensa = entorno.Paso(accion)
            
            agente.Guarda(estado, accion, recompensa, estado_siguiente)
            agente.Entrena()
            
            rn_total.append(entorno.rn)
            ln_total.append(entorno.ln)
            rc_total.append(entorno.rc)
            lc_total.append(entorno.lc)
            vc_total.append(entorno.vc)
            f_total.append(entorno.f)
            recompensa_total.append(entorno.recompensa)
            estado = estado_siguiente
        
        agente.Actualiza_Episodio()
        
        print('Resultado final: [{:.2f}, {:.2f}, {:.2f}, {:.2f}], vc = {:.2e}, f = {:.2e}'. \
                format(entorno.rn, entorno.ln, entorno.rc, entorno.lc,
                       entorno.vc, entorno.f))
        Muestra_Resultado(
            rn_total, ln_total, rc_total, lc_total,
            vc_total, f_total, recompensa_total, e)
        print()
        
    return
        

def Ejecuta_Deterministico() :
    
    #estado = entorno.Reinicia()
    estado = entorno.Reinicia_Aleatorio()
    rn_total = [entorno.rn]
    ln_total = [entorno.ln]
    rc_total = [entorno.rc]
    lc_total = [entorno.lc]
    vc_total = [entorno.vc]
    f_total  = [entorno.f]
    recompensa_total = [entorno.recompensa]
    estado = np.reshape(estado, (1, ESPACIO_ESTADOS))
    
    for p in range(PASOS_POR_EPISODIO) :
        
        accion = agente.Actua(estado, deterministico = True)
        estado_siguiente, recompensa = entorno.Paso(accion)
        estado_siguiente = np.reshape(estado_siguiente, (1, ESPACIO_ESTADOS))
        
        rn_total.append(entorno.rn)
        ln_total.append(entorno.ln)
        rc_total.append(entorno.rc)
        lc_total.append(entorno.lc)
        vc_total.append(entorno.vc)
        f_total.append(entorno.f)
        recompensa_total.append(entorno.recompensa)
        estado = estado_siguiente
    
    Muestra_Resultado(
        rn_total, ln_total, rc_total, lc_total,
        vc_total, f_total, recompensa_total, None)
    
    return estado, entorno.f
    
if __name__ == "__main__" :
    
    set_random_seed(1)
    
    entorno = Resonador()
    agente = Agente(RN_CAPAS)
    
    Entrena_Agente(EPISODIOS)
    agente.Guardar_Modelo('resultados/rl_resonador.h5')
    
    estado, f = Ejecuta_Deterministico()
    print('Resultado final: [{:.2f}, {:.2f}, {:.2f}, {:.2f}], vc = {:.2e}, f = {:.2e}'. \
        format(entorno.rn, entorno.ln, entorno.rc, entorno.lc,
                entorno.vc, entorno.f))
    
        
        
    