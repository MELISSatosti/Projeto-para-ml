# Projeto-para-ml
Classificação de métrica 
from sklearn.svm import LinearSVC
import numpy as np

# Define treino_x and treino_y with your training data
# Example data (replace with your actual data)
treino_x = np.array([[1, 2], [3, 4], [5, 6]]) # Example features (3 samples, 2 features)
treino_y = np.array([0, 1, 0]) # Example labels (3 samples)


model = LinearSVC()
model.fit(treino_x, treino_y)
pessoa_misteriosa = [ 0 , 0 ]
# Propositalmente coloquei as características de homem, então o
# resultado esperado deve ser 1 = Homem

model.predict([pessoa_misteriosa])
pessoa_misterio1 = [ 1 , 1 ] # Resultado esperado 0 = Mulher (Update features to match training data)
pessoa_misterio2 = [ 1 , 0 ] # Resultado dele é errar, forneci features de um homem que não treinei no modelo (Update features to match training data)
pessoa_misterio3 = [ 1 , 0 ] # Resultado esperado 1 = Homem (Update features to match training data)

# Agrupar todas as características dentro de uma variável
pessoas_misteriosas = [pessoa_misterio1, pessoa_misterio2, pessoa_misterio3]
sklearn.metrics import accuracy_score
# testes_y é classificação real feita por mim, servira para
# medir se acurácia do modelo, no processo de criação do modelo
# isso é de grande importância
testes_y = [ 0 , 1 , 1 ]
Homem1 = [ 0 , 0 , 1 , 0 , 1 ]
Homem2 = [ 0 , 0 , 1 , 0 , 0 ]
Homem3 = [ 1 , 0 , 0 , 1 , 0 ]
Homem4 = [ 1 , 0 , 1 , 1 , 0 ]
Homem5 = [ 1 , 0 , 1 , 0 , 0 ]

Mulher1 = [ 1 , 1 , 0 , 1 , 1 ]
Mulher2 = [ 1 , 0 , 0 , 1 , 1 ]
Mulher3 = [ 1 , 1 , 1 , 1 , 1 ]
Mulher4 = [ 0 , 1 , 1 , 0 , 1 ]
Mulher5 = [ 1 , 0 , 1 , 0 , 0 ]

treino_x_v2 = [Homem1, Homem2, Homem3, Homem4, Homem5, Mulher1, Mulher2, Mulher3, Mulher4, Mulher5]

treino_y_v2 = [ 1 , 1 , 1 , 1 , 1 , 0 , 0 , 0 , 0 , 0 ]

model_2 = LinearSVC()
model_2.fit(treino_x_v2, treino_y_v2)

pessoa_misterio1 = [ 1 , 1 , 0 , 1 , 1 ] # Resultado esperado 0 = Mulher
pessoa_misterio2 = [ 1 , 1 , 0 , 1 , 0 ] # Resultado dele é errar, forneci
# features de um homem que não treinei no modelo
pessoa_misterio3 = [ 0 , 1 , 1 , 0 , 1 ] # Resultado esperado 0 = Mulher
pessoa_misterio4 = [ 1 , 0, 1 , 0 , 0 ] # Resultado esperado 1 = Homem
pessoa_misterio5 = [ 1 , 0 , 1 , 0 , 0 ] # Resultado esperado 1 = Homem
pessoa_misterio6 = [ 0 , 1 , 1 , 0 , 1 ] # Resultado esperado 0 = Mulher
pessoa_misterio7 = [ 1 , 0 , 1 , 0 , 0 ] # Resultado esperado 1 = Homem

pessoas_misteriosas_v2 = [pessoa_misterio1, pessoa_misterio2, pessoa_misterio3, pessoa_misterio4, pessoa_misterio5, pessoa_misterio6
, pessoa_misterio7]

testes_y_v2 = [ 0 , 1 , 0 , 1 , 1 , 0 , 1 ]

previsoes_v2 = model_2.predict(pessoas_misteriosas_v2)

taxa_de_acerto_v2 = accuracy_score(testes_y_v2, previsoes_v2)
print ( "taxa de acertos v2 = %.2f" % (taxa_de_acerto_v2 * 100 ))

