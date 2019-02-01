import math
import numpy as np
import random

class Centroide:
    def __init__(self,color, coordenadas):
        #El color va a ser el identificador del centroide
        self.color = color
        self.coordenadas = coordenadas
        self.elementos = []
    #Metodo para poder calcular el promedio de todos los datos y redefinir centroide
    def promedioNuevoCentroide(self):
        for i in range(0,len(self.coordenadas)):
            suma = 0
            for a in range(0,len(self.elementos)):
                suma = suma + self.elementos[a][i]
            try:
                promedio = float(suma) / float(len(self.elementos))
            except:
                promedio = 0
            self.coordenadas[i] = promedio
        self.elementos = []
class KMeansImplementacion:
    #Metodo constructor. colores tiene que ser una matriz de la misma longitud que el numero de centroides
    def __init__(self, numeroClusters, colores, ejecuciones = 20):
        self.numeroClusters = numeroClusters
        self.ejecuciones = ejecuciones
        self.arrayConClusters = []
        self.colores = colores
    #Metodo para calcular la distancia euclediana entre el centroide de la clase y el dato que se tenga
    def distancia_euclediana(self,centroide,elemento):
      valores = []
      for i in range(0,len(centroide)):
        valores.append((centroide[i] - elemento[i])**2)
      suma = 0
      for i in range(0,len(valores)):
        suma = suma + valores[i]
      distancia = math.sqrt(suma)
      return distancia
    #Metodo entrenar que tomara como parametros de entrada los datos
    def entrenar(self, data):
        #Cuerpo de codigo que inicializara los centroides aleatoriamente
        for i in range(0, self.numeroClusters):
            self.arrayConClusters.append(Centroide(self.colores[i], data[int(random.random() * len(data)) - 1]))
        #Numero de ejecuciones para entrenar el algoritmo KMeans
        for u in range(0, self.ejecuciones):
            #Iterar sobre todos los elementos de los datos que se ingresen
            for i in range(0, len(data)):
                distancias = []
                #Ciclo para poder calcular las distancias del punto con respecto al centroide para obtener la menor distancia
                for a in range(0,len(self.arrayConClusters)):
                    distancia = self.distancia_euclediana(self.arrayConClusters[a].coordenadas, data[i])
                    distancias.append(distancia)
                #Aqui tengo las distancias del elemento data[i] con cada ckuster que hay
                #Anexamos el elemento al cluster con respecto al cual tenga menor distancia
                self.arrayConClusters[distancias.index(min(distancias))].elementos.append(data[i])
            #Matriz que contiene los elementos de los clusters
            self.matrizElementos = []
            for u in range(0,len(self.arrayConClusters)):
                self.matrizElementos.append(self.arrayConClusters[u].elementos)
            #Ahora redefiniremos la posicion de los clusters
            for i in range(0,len(self.arrayConClusters)):
                self.arrayConClusters[i].promedioNuevoCentroide()
    #Metodo que te dira a que grupo/cluster pertenece el dato que quieras
    def predecir(self, data):
        distancias = []
        for i in range(0,len(self.arrayConClusters)):
            distancia = self.distancia_euclediana(self.arrayConClusters[i].coordenadas, data)
            distancias.append(distancia)
        return self.arrayConClusters[distancias.index(min(distancias))].color
#Funcion para convertir los datos alfabeticos a numericos
def convertir_dataNoNumerico(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
            df[column] = list(map(convert_to_int,df[column]))
    return df
