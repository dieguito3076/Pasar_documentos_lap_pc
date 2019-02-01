import random
from AlgoritmoGeneticoGeneral import Genetic_Algorithm
import numpy as np
#Funcion para crear numero binario en cada elemento de la poblacion
def numero_binario():
    numero = random.random()
    if(numero < 0.5):
        return 0
    else:
        return 1
#Funcion para crear la pobracion
def crear_poblacion(numero_de_poblacion_inicial, len_elemento):
    poblacion = []
    for i in range(0,numero_de_poblacion_inicial):
        numero = str(numero_binario())
        for i in range(0, len_elemento-1):
            numero = numero + str(numero_binario())
        poblacion.append(numero)
    return poblacion


poblacion = crear_poblacion(6, 3)
print poblacion
#AG = Genetic_Algorithm(poblacion, '101', 0.01, 0.3,'cross_over_probabilistic')


#print AG.population
#AG.evolve()
#print AG.population
