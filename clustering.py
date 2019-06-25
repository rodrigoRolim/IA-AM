# -*- coding: utf-8 -*-
import json
import pandas as pd
from math import sqrt
import numpy as np
import plotly.io as pio
import plotly.offline as py
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.io.arff import loadarff 
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# py.init_notebook_mode(connected=False)

def optimal_number_of_clusters(wcss):
  x1, y1 = 2, wcss[0]
  x2, y2 = 15, wcss[len(wcss)-1]
 
  distances = []
  for i in range(len(wcss)):
      x0 = i+2
      y0 = wcss[i]
      numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
      denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
      distances.append(numerator/denominator)

  return distances.index(max(distances)) + 2
def writeJSONFile(data):
  filePathNameWExt = 'results_2.json'
  with open(filePathNameWExt, 'a') as fp:
    json.dump(data, fp)



listFile = ['vegetables/FCTH.arff']
# Carrega o .arff
datastore = []
for selectedNormalization in range(1, 5):

  for item in listFile:
    
    name = item.split('/')
    descritor = name[1].split('.')[0]
    
    raw_data = loadarff(item)
    # Transforma o .arff em um Pandas Dataframe
    df = pd.DataFrame(raw_data[0])
    # Imprime o Dataframe com suas colunas
    df.head()
    # Com o iloc voce retira as linhas e colunas que quiser do Dataframe, no caso aqui sem as classes
    data_aux = df.iloc[:, 0:-1].values
    print(data_aux)

    # Escolha umas das 4 técnicas de normalização existentes
    # 1 = MinMaxScaler, 2 = StandardScaler, 3 = MaxAbsScaler, 4 = RobustScaler

    if selectedNormalization == 1:
      scaler = preprocessing.MinMaxScaler()
    if selectedNormalization == 2:
      scaler = preprocessing.StandardScaler()
    if selectedNormalization == 3:
      scaler = preprocessing.MaxAbsScaler()
    if selectedNormalization == 4:
      scaler = preprocessing.RobustScaler()
      
    data_normalized = scaler.fit_transform(data_aux)
    print(data_normalized)

    # Chamando a função do gráfico interativo
   #  configure_plotly_browser_state()

    # Lista para armazenar os valores dados pelo método
    elbow = []
    # Numpy Array de valores para variar de 2 a 11 para servir de X no gráfico
    clusters = np.arange(2, 16)

    # Loop variando de 2 a 11 clusteres
    best_k = 0
    for i in range(2, 16):
      # Inicializando KMeans com I Clusters
      kmeans = KMeans(n_clusters = i, init = 'random')
      # Agrupa os dados
      kmeans.fit(data_normalized)
      # Imprime na tela
      print('Elbow Method: Número de Cluster - {} Valor - {}'.format(i, kmeans.inertia_))
      # Armazena na lista os valores dado pelo método
      elbow.append(kmeans.inertia_)
      print('')
        
    # Plotando o gráfico de pontos + linha
    dados = go.Scatter(
        # Eixo x recebe o array de clusters
        x = clusters,
        # Eixo y recebe os valores do método
        y = elbow,
        # Define que o gráfico será de pontos e linhas
        mode = 'lines+markers',
        # Customina os marcadores
        marker = dict(
            # Define a cor
            color = '#36bce2',
            # Tamanho dos pontos
            size = 10,
            # Tamanho da linha ou contorno
            line = dict(width = 3)
        ),
    )

    # Alterando configurações de Layout do Gráfico
    layout = go.Layout(
        # Define Título
        title = 'Método Elbow',
        # Define o nome do eixo X
        xaxis = {'title': 'Número de Clusters'},
        # Define o nome do eixo Y
        yaxis = {'title':'Soma dos quadrados'},
        # Define a cor da borda e contorno do gráfico
        paper_bgcolor='rgba(245, 246, 249, 1)',
        # Define a cor do fundo do gráfico
        plot_bgcolor='rgba(245, 246, 249, 1)'
    )
    best_k = optimal_number_of_clusters(elbow)
    print(elbow)
    # Plotando
    data = [dados]
    i = str(selectedNormalization)
    fig = go.Figure(data=data, layout=layout)
    pio.write_image(fig, 'images_clustering/'+i+name[0]+name[1]+'_e'+'.png')

    #Aplicando PCA para descobrir as 2 melhores features
    pca = PCA(n_components = 2)
    bestFeatures = pca.fit_transform(data_normalized)

    # Quando diminuimos diversas dimensões em apenas duas, perdemos muitas informações;
    # Com esse atributo podemos ver qual a porcentagem de informação foi perdida.
    print('VARIÂNCIA DAS MELHORES FEATURES')
    print(pca.explained_variance_ratio_)
    print('')
    
    # ALGUNS PARÂMETROS DO MÉTODO KMEANS

    # A variável n_clusters determina o número de agrupamento que queremos gerar com os dados;
    # Esse número normalmente é o número de classes, mas devido a complexidade de alguns conjuntos de dados talvez ele não seja o ideal;
    # Existem técnicas para descobrir o valor K ideal para o dataset, assim como fizemos acima.

    # O parametro init se refere ao modo como o algoritmo será inicializado.
    # A lib Scikit-Learn nos fornece três métodos de inicialização, sendo eles:
    # k-means++: É o método padrão, e os centróides serão gerados utilizando um método inteligente que favorece a convergência.
    # random: Inicializa de forma aleatória, ou seja, os centróides iniciais serão gerados de forma totalmente aleatória sem um critério para seleção.
    # ndarray: Esse podemos especificar um array de valores indicando qual seriam os centróides que o algoritmo deveria utilizar para a inicialização.

    # O parâmetro max_iter se refere a quantidade máxima de vezes que o algoritmo irá executar, por padrão o valor é 300 iterações.

    # Inicializa o KMeans com 3 clusters e modo de inicialização aleatória.
    parameters_k = ['k-means++', 'random']
    for j in range(2):
      kmeans = KMeans(n_clusters = best_k, init = parameters_k[j])

      # Aqui de fato agrupamos os dados que queremos.
      kmeans.fit(bestFeatures)

      # Nesta variável já temos as coordenadas dos centróides gerados pelo agrupamento.
      print('COORDENADA DOS CENTROIDES')
      print('')
      print(kmeans.cluster_centers_)
      print('')

      # Uma funcionalidade interessante é a função fit_transform;
      # Ela executa o K-means para agrupar os dados e retorna uma tabela de distâncias;
      # É gerada de forma que em cada instância contém os valores de distância em relação a cada cluster.
      distance = kmeans.fit_transform(bestFeatures)
      print('TABELA DE DISTÂNCIAS')
      print('')
      print(distance)
      print('')

      # O atributo labels_ nos retorna os labels para cada instância, ou seja, o número do cluster que a instância de dados foi atribuída.
      labels = kmeans.labels_
      # Já o predict serve para testar novas amostras e ver a qual clusters elas são atribuidas
      preds = kmeans.predict(bestFeatures)
      print('RÓTULO DADO PELO KMEANS DAS AMOSTRAS')
      print('')
      print(labels)
      print(preds)
      print('')
      result = {
        "best_k": best_k,
        "normalization": selectedNormalization,
        "descritor": descritor,
        "dataset": name[0],
        "parametro": parameters_k[j],
        "variancia": pca.explained_variance_ratio_.tolist()
      }
      datastore.append(result)
      # Chamando a função do gráfico interativo
      # configure_plotly_browser_state()

      # Plotando o gráfico das Amostras
      dados1 = go.Scatter(
          # COM PCA
          # Eixo x 
          x = bestFeatures[:, 0],
          # Eixo y 
          y = bestFeatures[:, 1],
          # SEM PCA
          # Eixo x 
          #x = data_normalized[:, 0],
          # Eixo y 
          #y = data_normalized[:, 1],
          # Define que o gráfico será de pontos
          mode = 'markers',
          # Define o nome que será impresso na legenda do gráfico
          name = 'Instâncias',
          # Customina os marcadores
          marker = dict(
              # Define a cor
              color = preds,
              # Define a paleta de cores
              colorscale = 'Portland',
              # Tamanho dos pontos
              size = 10,
              # Opacidade dos pontos
              opacity = 1.0,
              # Tamanho da linha ou contorno
              line = dict(width = 1)
          ),
          showlegend = False
      )

      # Plotando o gráfico dos Centróides
      dados2 = go.Scatter(
          # Eixo x 
          x = kmeans.cluster_centers_[:, 0],
          # Eixo y 
          y = kmeans.cluster_centers_[:, 1],
          # Define que o gráfico será de pontos
          mode = 'markers',
          # Define o nome que será impresso na legenda do gráfico
          name = 'Clusters',
          # Customina os marcadores
          marker = dict(
              # Define a cor
              color = 'black',
              # Tamanho dos pontos
              size = 20,
              # Opacidade dos pontos
              opacity = 1.0,
              # Tamanho da linha ou contorno
              line = dict(width = 2)
          ),
      )

      # Alterando configurações de Layout do Gráfico
      layout = go.Layout(
          # Define Título
          title = 'Clusterização',
          # Define o nome do eixo X
          #xaxis = {'title': 'Número de Clusters'},
          # Define o nome do eixo Y
          #yaxis = {'title':'Soma dos quadrados'},
          # Define a cor da borda e contorno do gráfico
          paper_bgcolor='rgba(245, 246, 249, 1)',
          # Define a cor do fundo do gráfico
          plot_bgcolor='rgba(245, 246, 249, 1)'
      )

      # Plotando
      data = [dados1, dados2]
      fig = go.Figure(data=data, layout=layout)
      pio.write_image(fig, 'images_clustering/'+i+parameters_k[j]+name[0]+name[1]+'_p'+'.png')

      print(best_k)
writeJSONFile(datastore)