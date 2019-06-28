# -*- coding: utf-8 -*-
import json
import pandas as pd
import numpy as np
import plotly.io as pio
import plotly.offline as py
import plotly.graph_objs as go
from sklearn import preprocessing
from scipy.io.arff import loadarff
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import time
import warnings
warnings.filterwarnings("ignore")

def writeJSONFile(data):
  filePathNameWExt = 'results_active.json'
  with open(filePathNameWExt, 'a') as fp:
    json.dump(data, fp)

# Criando as variáveis do Aprendizado Ativo fora do ciclo de iteração
active_data = []
active_label = []
# Transformando as listas em Numpy Array
active_data = np.asarray(active_data)
active_label = np.asarray(active_label)

# Criando lista de acurácias global
acc_train_geral = []
acc_test_geral = []

# Lista para controlar o num de iterações
iteracoes = []

listFile = ['scissors/PHOG.arff', 'shapes/PHOG.arff']
datastore = [] 
for selectedNormalization in range(1,3):
  
  for item in listFile:
    for num_c in range(3, 8):
      # Carrega o .arff
      raw_data = loadarff(item)
      name = item.split('/')
      descritor = name[1].split('.')[0]
      # Transforma o .arff em um Pandas Dataframe
      df = pd.DataFrame(raw_data[0])
      # Imprime o Dataframe com suas colunas
      df.head()

      # Com o iloc voce retira as linhas e colunas que quiser do Dataframe, no caso aqui sem as classes
      X = df.iloc[:, 0:-1].values

      # Aqui salvamos apenas as classes agora
      y = df['class']
      # Substituimos os valores binários por inteiro
      bow = []
      int_value = 0
      y_aux = []
      for i in y:
        if i in bow:
          y_aux.append(int_value)
        else:
          bow.append(i)
          int_value += 1
          y_aux.append(int_value)
      # Novo y
      y = y_aux

      # Dividindo o conjunto em 80% Treino e 20% Teste.
      # O parâmetro random_state = 327 define que sempre será dividido da mesma forma o conjunto.
      X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=327)

      print('Tamanho do conjunto de Treino: {}'.format(X_train.shape))
      print('Tamanho do conjunto de Teste: {}'.format(X_test.shape))

      # Escolha umas das 4 técnicas de normalização existentes
      # 1 = MinMaxScaler, 2 = StandardScaler, 3 = MaxAbsScaler, 4 = RobustScaler
      # selectedNormalization = 1]

      if selectedNormalization == 1:
        scaler = preprocessing.MinMaxScaler()
      if selectedNormalization == 2:
        scaler = preprocessing.StandardScaler()
      if selectedNormalization == 3:
        scaler = preprocessing.MaxAbsScaler()
      if selectedNormalization == 4:
        scaler = preprocessing.RobustScaler()
        
      # Escalando os dados de treinamento
      X_train = scaler.fit_transform(X_train)
      # Escalando os dados de teste com os dados de treinamento, visto que os dados de teste podem ser apenas 1 amostra
      X_test = scaler.transform(X_test)

      print('Média do Conjunto de Treinamento por Feature:')
      print(X_train.mean(axis = 0))
      print('Desvio Padrão do Conjunto de Treinamento por Feature:')
      print(X_train.std(axis = 0))

      # Aplicando K-Means para separar as amostras não rotuladas em clusters

      # Inicializar o KMeans com N centroides
      # num_c = 7
      kmeans = KMeans(n_clusters = int(num_c), init = 'random')
      print('Numero de clusters: {}'.format(int(num_c)))
      print('')
      # Executar passando como parâmetro os dados
      kmeans.fit(X_train)
      # Variavel centers recebe os centroides gerados que possuem o valor de cada dimensao onde o centroide esta localizado
      centers = kmeans.cluster_centers_
      #print(centers)
      # Variavel distance recebe uma tabela de distancia de cada amostra para o centroide
      distance = kmeans.fit_transform(X_train)
      #print(distance)
      # Aqui selecionamos as amostras mais próximas de cada centróide para utilizar como amostras raízes
      root_index = np.argmin(distance, axis=0)
      root_index = np.sort(root_index)[::-1]
      print('Amostras selecionadas como Raízes:')
      print(root_index)
      amostras_raizes = []
      amostras_raizes_labels = []
      rotulos = kmeans.labels_
      for i in root_index:
        amostras_raizes.append(X_train[i])
        amostras_raizes_labels.append(y_train[i])
        X_train = np.delete(X_train, i, 0)
        y_train = np.delete(y_train, i, 0)
        rotulos = np.delete(rotulos, i, 0)
        
      # Transforma elas em ndarray novamente
      amostras_raizes = np.asarray(amostras_raizes)
      amostras_raizes_labels = np.asarray(amostras_raizes_labels)

      # Variável para controlar a primeira iteração
      firstIteration = True

      # Transformar os ndarray que armazenaram as amostras selecionadas em listas para usar o método append
      active_data = active_data.tolist()
      active_label = active_label.tolist()

      # Selecionando amostras a partir do agrupamento
      if firstIteration:
        for i in amostras_raizes:
          active_data.append(i)
        for i in amostras_raizes_labels:
          active_label.append(i)
        firstIteration = False
      else:
        for i in range(0, num_c):
          sample = list(rotulos).index(i)
          rotulos = np.delete(rotulos, i, 0)
          active_data.append(X_train[sample])
          active_label.append(y_train[sample])
          X_train = np.delete(X_train, sample, 0)
          y_train = np.delete(y_train, sample, 0)

      # Transforma elas em ndarray novamente
      active_data = np.asarray(active_data)
      active_label = np.asarray(active_label)

      # Inicializar os classificadores

      # Gaussian Naive Bayes
      t = time.time()
      gnb = GaussianNB()
      model1 = gnb.fit(active_data, active_label)
      print('Treino do Gaussian Naive Bayes Terminado. (Tempo de execucao: {})'.format(time.time() - t))

      # Logistic Regression
      t = time.time()
      logreg = LogisticRegression()
      model2 = logreg.fit(active_data, active_label)
      print('Treino do Logistic Regression Terminado. (Tempo de execucao: {})'.format(time.time() - t))

      # Decision Tree
      t = time.time()
      dectree = DecisionTreeClassifier()
      model3 = dectree.fit(active_data, active_label)
      print('Treino do Decision Tree Terminado. (Tempo de execucao: {})'.format(time.time() - t))

      # K-Nearest Neighbors
      t = time.time()
      knn = KNeighborsClassifier(n_neighbors = 3)
      model4 = knn.fit(active_data, active_label)
      print('Treino do K-Nearest Neighbors Terminado. (Tempo de execucao: {})'.format(time.time() - t))
        
      # Linear Discriminant Analysis
      t = time.time()
      lda = LinearDiscriminantAnalysis()
      model5 = lda.fit(active_data, active_label)
      print('Treino do Linear Discriminant Analysis Terminado. (Tempo de execucao: {})'.format(time.time() - t))

      # Support Vector Machine
      t = time.time()
      svm = SVC()
      model6 = svm.fit(active_data, active_label)
      print('Treino do Support Vector Machine Terminado. (Tempo de execucao: {})'.format(time.time() - t))

      # RandomForest
      t = time.time()
      rf = RandomForestClassifier()
      model7 = rf.fit(active_data, active_label)
      print('Treino do RandomForest Terminado. (Tempo de execucao: {})'.format(time.time() - t))

      # Neural Net
      t = time.time()
      nnet = MLPClassifier(alpha=1)
      model8 = nnet.fit(active_data, active_label)
      print('Treino do Neural Net Terminado. (Tempo de execucao: {})'.format(time.time() - t))
      print('')

      # Cria 2 vetores de predicoes para armazenar todas acuracias e outros para as métricas
      acc_train = []
      acc_test = []
      f1score = []
      precision = []
      recall = []

      # Gaussian Naive Bayes

      # Variavel para armazenar o tempo
      t = time.time()
      # Usando o modelo para predição das amostras de teste
      aux = gnb.predict(X_test)
      # Método para criar a matriz de confusão
      cm = confusion_matrix(y_test, aux)
      # Método para calcular o valor F1-Score
      f1score.append(f1_score(y_test, aux, average = 'macro'))
      # Método para calcular a Precision
      precision.append(precision_score(y_test, aux, average = 'macro'))
      # Método para calcular o Recall
      recall.append(recall_score(y_test, aux, average = 'macro'))
      # Salvando as acurácias nas listas
      acc_train.append(gnb.score(X_train, y_train))
      acc_test.append(gnb.score(X_test, y_test))
      #print('Acuracia obtida com o Gaussian Naive Bayes no Conjunto de Treinamento: {:.2f}'.format(acc_train[0]))
      #print('Acuracia obtida com o Gaussian Naive Bayes no Conjunto de Teste: {:.2f}'.format(acc_test[0]))
      #print('Matriz de Confusão:')
      #print(cm)
      #print('Precision: {:.5f}'.format(precision[0]))
      #print('Recall: {:.5f}'.format(recall[0]))
      #print('F1-score: {:.5f}'.format(f1score[0]))
      #print('(Tempo de execucao: {:.5f})'.format(time.time() - t))
      print('')

      data = {
        'normalization': selectedNormalization,
        'date': name[0],
        'extrator': descritor,
        'classifier': 'Gaussian Naive Bayes',
        'treinamento': '{:.2f}'.format(acc_train[0]), 
        'teste': '{:.2f}'.format(acc_test[0]),
        'precision': '{:.5f}'.format(precision[0]),
        'recall': '{:.5f}'.format(recall[0]),
        'f1_score': '{:.5f}'.format(f1score[0]),
        'runningtime': '{:.5f}'.format(time.time() - t)
      }
      datastore.append(data)
      # Logistic Regression

      # Variavel para armazenar o tempo
      t = time.time()
      # Usando o modelo para predição das amostras de teste
      aux = logreg.predict(X_test)
      # Método para criar a matriz de confusão
      cm = confusion_matrix(y_test, aux)
      # Método para calcular o valor F1-Score
      f1score.append(f1_score(y_test, aux, average = 'macro'))
      # Método para calcular a Precision
      precision.append(precision_score(y_test, aux, average = 'macro'))
      # Método para calcular o Recall
      recall.append(recall_score(y_test, aux, average = 'macro'))
      # Salvando as acurácias nas listas
      acc_train.append(logreg.score(X_train, y_train))
      acc_test.append(logreg.score(X_test, y_test))
      #print('Acuracia obtida com o Logistic Regression no Conjunto de Treinamento: {:.2f}'.format(acc_train[1]))
      #print('Acuracia obtida com o Logistic Regression no Conjunto de Teste: {:.2f}'.format(acc_test[1]))
      #print('Matriz de Confusão:')
      #print(cm)
      #print('Precision: {:.5f}'.format(precision[1]))
      #print('Recall: {:.5f}'.format(recall[1]))
      #print('F1-score: {:.5f}'.format(f1score[1]))
      #print('(Tempo de execucao: {:.5f})'.format(time.time() - t))
      print('')

      data = {
        'normalization': selectedNormalization,
        'date': name[0],
        'extrator': descritor,
        'classifier':'Logistic Regression',
        'treinamento': '{:.2f}'.format(acc_train[1]), 
        'teste': '{:.2f}'.format(acc_test[1]),
        'precision': '{:.5f}'.format(precision[1]),
        'recall': '{:.5f}'.format(recall[1]),
        'f1_score': '{:.5f}'.format(f1score[1]),
        'runningtime': '{:.5f}'.format(time.time() - t)
      }
      datastore.append(data)
      # Decision Tree

      # Variavel para armazenar o tempo
      t = time.time()
      # Usando o modelo para predição das amostras de teste
      aux = dectree.predict(X_test)
      # Método para criar a matriz de confusão
      cm = confusion_matrix(y_test, aux)
      # Método para calcular o valor F1-Score
      f1score.append(f1_score(y_test, aux, average = 'macro'))
      # Método para calcular a Precision
      precision.append(precision_score(y_test, aux, average = 'macro'))
      # Método para calcular o Recall
      recall.append(recall_score(y_test, aux, average = 'macro'))
      # Salvando as acurácias nas listas
      acc_train.append(dectree.score(X_train, y_train))
      acc_test.append(dectree.score(X_test, y_test))
      #print('Acuracia obtida com o Decision Tree no Conjunto de Treinamento: {:.2f}'.format(acc_train[2]))
      #print('Acuracia obtida com o Decision Tree no Conjunto de Teste: {:.2f}'.format(acc_test[2]))
      #print('Matriz de Confusão:')
      #print(cm)
      #print('Precision: {:.5f}'.format(precision[2]))
      #print('Recall: {:.5f}'.format(recall[2]))
      #print('F1-score: {:.5f}'.format(f1score[2]))
      #print('(Tempo de execucao: {:.5f})'.format(time.time() - t))
      print('')
      data = {
        'normalization': selectedNormalization,
        'date': name[0],
        'extrator': descritor,
        'classifier':'Decision Tree',
        'treinamento': '{:.2f}'.format(acc_train[2]), 
        'teste': '{:.2f}'.format(acc_test[2]),
        'precision': '{:.5f}'.format(precision[2]),
        'recall': '{:.5f}'.format(recall[2]),
        'f1_score': '{:.5f}'.format(f1score[2]),
        'runningtime': '{:.5f}'.format(time.time() - t)
      }
      datastore.append(data)
      # K-Nearest Neighbors

      # Variavel para armazenar o tempo
      t = time.time()
      # Usando o modelo para predição das amostras de teste
      aux = knn.predict(X_test)
      # Método para criar a matriz de confusão
      cm = confusion_matrix(y_test, aux)
      # Método para calcular o valor F1-Score
      f1score.append(f1_score(y_test, aux, average = 'macro'))
      # Método para calcular a Precision
      precision.append(precision_score(y_test, aux, average = 'macro'))
      # Método para calcular o Recall
      recall.append(recall_score(y_test, aux, average = 'macro'))
      # Salvando as acurácias nas listas
      acc_train.append(knn.score(X_train, y_train))
      acc_test.append(knn.score(X_test, y_test))
      #print('Acuracia obtida com o K-Nearest Neighbors no Conjunto de Treinamento: {:.2f}'.format(acc_train[3]))
      #print('Acuracia obtida com o K-Nearest Neighbors no Conjunto de Teste: {:.2f}'.format(acc_test[3]))
      #print('Matriz de Confusão:')
      #print(cm)
      #print('Precision: {:.5f}'.format(precision[3]))
      #print('Recall: {:.5f}'.format(recall[3]))
      #print('F1-score: {:.5f}'.format(f1score[3]))
      #print('(Tempo de execucao: {:.5f})'.format(time.time() - t))
      print('')
      
      data = {
        'normalization': selectedNormalization,
        'date': name[0],
        'extrator': descritor,
        'classifier':'K-Nearest Neighbors',
        'treinamento': '{:.2f}'.format(acc_train[3]), 
        'teste': '{:.2f}'.format(acc_test[3]),
        'precision': '{:.5f}'.format(precision[3]),
        'recall': '{:.5f}'.format(recall[3]),
        'f1_score': '{:.5f}'.format(f1score[3]),
        'runningtime': '{:.5f}'.format(time.time() - t)
      }
      datastore.append(data)
      # Linear Discriminant Analysis

      # Variavel para armazenar o tempo
      t = time.time()
      # Usando o modelo para predição das amostras de teste
      aux = lda.predict(X_test)
      # Método para criar a matriz de confusão
      cm = confusion_matrix(y_test, aux)
      # Método para calcular o valor F1-Score
      f1score.append(f1_score(y_test, aux, average = 'macro'))
      # Método para calcular a Precision
      precision.append(precision_score(y_test, aux, average = 'macro'))
      # Método para calcular o Recall
      recall.append(recall_score(y_test, aux, average = 'macro'))
      # Salvando as acurácias nas listas
      acc_train.append(lda.score(X_train, y_train))
      acc_test.append(lda.score(X_test, y_test))
      #print('Acuracia obtida com o Linear Discriminant Analysis no Conjunto de Treinamento: {:.2f}'.format(acc_train[4]))
      #print('Acuracia obtida com o Linear Discriminant Analysis no Conjunto de Teste: {:.2f}'.format(acc_test[4]))
      #print('Matriz de Confusão:')
      #print(cm)
      #print('Precision: {:.5f}'.format(precision[4]))
      #print('Recall: {:.5f}'.format(recall[4]))
      #print('F1-score: {:.5f}'.format(f1score[4]))
      #print('(Tempo de execucao: {:.5f})'.format(time.time() - t))
      print('')
      data = {
        'normalization': selectedNormalization,
        'date': name[0],
        'extrator': descritor,
        'classifier':'Linear Discriminant Analysis',
        'treinamento': '{:.2f}'.format(acc_train[4]), 
        'teste': '{:.2f}'.format(acc_test[4]),
        'precision': '{:.5f}'.format(precision[4]),
        'recall': '{:.5f}'.format(recall[4]),
        'f1_score': '{:.5f}'.format(f1score[4]),
        'runningtime': '{:.5f}'.format(time.time() - t)
      }
      datastore.append(data)
      # Support Vector Machine

      # Variavel para armazenar o tempo
      t = time.time()
      # Usando o modelo para predição das amostras de teste
      aux = svm.predict(X_test)
      # Método para criar a matriz de confusão
      cm = confusion_matrix(y_test, aux)
      # Método para calcular o valor F1-Score
      f1score.append(f1_score(y_test, aux, average = 'macro'))
      # Método para calcular a Precision
      precision.append(precision_score(y_test, aux, average = 'macro'))
      # Método para calcular o Recall
      recall.append(recall_score(y_test, aux, average = 'macro'))
      # Salvando as acurácias nas listas
      acc_train.append(svm.score(X_train, y_train))
      acc_test.append(svm.score(X_test, y_test))
      #print('Acuracia obtida com o Support Vector Machine no Conjunto de Treinamento: {:.2f}'.format(acc_train[5]))
      #print('Acuracia obtida com o Support Vector Machine no Conjunto de Teste: {:.2f}'.format(acc_test[5]))
      #print('Matriz de Confusão:')
      #print(cm)
      #print('Precision: {:.5f}'.format(precision[5]))
      #print('Recall: {:.5f}'.format(recall[5]))
      #print('F1-score: {:.5f}'.format(f1score[5]))
      #print('(Tempo de execucao: {:.5f})'.format(time.time() - t))
      print('')
      
      data = {
        'normalization': selectedNormalization,
        'date': name[0],
        'extrator': descritor,
        'classifier':'Support Vector Machine',
        'treinamento': '{:.2f}'.format(acc_train[5]), 
        'teste': '{:.2f}'.format(acc_test[5]),
        'precision': '{:.5f}'.format(precision[5]),
        'recall': '{:.5f}'.format(recall[5]),
        'f1_score': '{:.5f}'.format(f1score[5]),
        'runningtime': '{:.5f}'.format(time.time() - t)
      }
      datastore.append(data)
      # RandomForest

      # Variavel para armazenar o tempo
      t = time.time()
      # Usando o modelo para predição das amostras de teste
      aux = rf.predict(X_test)
      # Método para criar a matriz de confusão
      cm = confusion_matrix(y_test, aux)
      # Método para calcular o valor F1-Score
      f1score.append(f1_score(y_test, aux, average = 'macro'))
      # Método para calcular a Precision
      precision.append(precision_score(y_test, aux, average = 'macro'))
      # Método para calcular o Recall
      recall.append(recall_score(y_test, aux, average = 'macro'))
      # Salvando as acurácias nas listas
      acc_train.append(rf.score(X_train, y_train))
      acc_test.append(rf.score(X_test, y_test))
      #print('Acuracia obtida com o RandomForest no Conjunto de Treinamento: {:.2f}'.format(acc_train[6]))
      #print('Acuracia obtida com o RandomForest no Conjunto de Teste: {:.2f}'.format(acc_test[6]))
      #print('Matriz de Confusão:')
      #print(cm)
      #print('Precision: {:.5f}'.format(precision[6]))
      #print('Recall: {:.5f}'.format(recall[6]))
      #print('F1-score: {:.5f}'.format(f1score[6]))
      #print('(Tempo de execucao: {:.5f})'.format(time.time() - t))
      print('')

      data = {
        'normalization': selectedNormalization,
        'date': name[0],
        'extrator': descritor,
        'classifier':'RandomForest',
        'treinamento': '{:.2f}'.format(acc_train[6]), 
        'teste': '{:.2f}'.format(acc_test[6]),
        'precision': '{:.5f}'.format(precision[6]),
        'recall': '{:.5f}'.format(recall[6]),
        'f1_score': '{:.5f}'.format(f1score[6]),
        'runningtime': '{:.5f}'.format(time.time() - t)
      }
      datastore.append(data)
      # Neural Net

      # Variavel para armazenar o tempo
      t = time.time()
      # Usando o modelo para predição das amostras de teste
      aux = nnet.predict(X_test)
      # Método para criar a matriz de confusão
      cm = confusion_matrix(y_test, aux)
      # Método para calcular o valor F1-Score
      f1score.append(f1_score(y_test, aux, average = 'macro'))
      # Método para calcular a Precision
      precision.append(precision_score(y_test, aux, average = 'macro'))
      # Método para calcular o Recall
      recall.append(recall_score(y_test, aux, average = 'macro'))
      # Salvando as acurácias nas listas
      acc_train.append(nnet.score(X_train, y_train))
      acc_test.append(nnet.score(X_test, y_test))
      #print('Acuracia obtida com o Neural Net no Conjunto de Treinamento: {:.2f}'.format(acc_train[7]))
      #print('Acuracia obtida com o Neural Net no Conjunto de Teste: {:.2f}'.format(acc_test[7]))
      #print('Matriz de Confusão:')
      #print(cm)
      #print('Precision: {:.5f}'.format(precision[7]))
      #print('Recall: {:.5f}'.format(recall[7]))
      #print('F1-score: {:.5f}'.format(f1score[7]))
      #print('(Tempo de execucao: {:.5f})'.format(time.time() - t))
      print('')

      data = {
        'normalization': selectedNormalization,
        'date': name[0],
        'extrator': descritor,
        'classifier':'Neural Net',
        'treinamento': '{:.2f}'.format(acc_train[7]), 
        'teste': '{:.2f}'.format(acc_test[7]),
        'precision': '{:.5f}'.format(precision[7]),
        'recall': '{:.5f}'.format(recall[7]),
        'f1_score': '{:.5f}'.format(f1score[7]),
        'runningtime': '{:.5f}'.format(time.time() - t)
      }
      datastore.append(data)
      # Salvando as acurácias globais
      acc_train_geral.append(acc_train)
      acc_test_geral.append(acc_test)

      # Chamando a função do gráfico interativo
      #  configure_plotly_browser_state()

      # Criando listas de acurácias por classificador
      acc_nb = []
      acc_lr = []
      acc_dt = []
      acc_knn = []
      acc_lda = []
      acc_svm = []
      acc_rf = []
      acc_nnet = []

      for i in acc_test_geral:
        acc_nb.append(i[0])
        acc_lr.append(i[1])
        acc_dt.append(i[2])
        acc_knn.append(i[3])
        acc_lda.append(i[4])
        acc_svm.append(i[5])
        acc_rf.append(i[6])
        acc_nnet.append(i[7])

      iteracoes = []
      for i in range(0, len(acc_test_geral)):
        iteracoes.append(i + 1)

      # Criando valores do eixo X
      eixo_x = ['GaussianNB', 'Logistic Regression', 'Decision Tree', 'k-NN', 'LDA', 'SVM', 'RandomForest', 'Neural Net']

      # Imprimindo o Número de Amostras utilizadas
      print('Número de Amostras selecionadas Ativamente: {}'.format(len(active_label)))
      print('')

      # Plotando os gráficos

      dados_nb = go.Scatter(
          # Eixo x recebe a iteração
          x = iteracoes,
          # Eixo y recebe os valores de acurácia
          y = acc_nb,
          # Define o nome
          name = 'GaussianNB',
          mode = 'lines+markers'
      )
      dados_lr = go.Scatter(
          # Eixo x recebe a iteração
          x = iteracoes,
          # Eixo y recebe os valores de acurácia
          y = acc_lr,
          # Define o nome
          name = 'Logistic Regression',
          mode = 'lines+markers'
      )
      dados_dt = go.Scatter(
          # Eixo x recebe a iteração
          x = iteracoes,
          # Eixo y recebe os valores de acurácia
          y = acc_dt,
          # Define o nome
          name = 'Decision Tree',
          mode = 'lines+markers'
      )
      dados_knn = go.Scatter(
          # Eixo x recebe a iteração
          x = iteracoes,
          # Eixo y recebe os valores de acurácia
          y = acc_knn,
          # Define o nome
          name = 'k-NN',
          mode = 'lines+markers'
      )
      dados_knn = go.Scatter(
          # Eixo x recebe a iteração
          x = iteracoes,
          # Eixo y recebe os valores de acurácia
          y = acc_knn,
          # Define o nome
          name = 'k-NN',
          mode = 'lines+markers'
      )
      dados_lda = go.Scatter(
          # Eixo x recebe a iteração
          x = iteracoes,
          # Eixo y recebe os valores de acurácia
          y = acc_lda,
          # Define o nome
          name = 'LDA',
          mode = 'lines+markers'
      )
      dados_svm = go.Scatter(
          # Eixo x recebe a iteração
          x = iteracoes,
          # Eixo y recebe os valores de acurácia
          y = acc_svm,
          # Define o nome
          name = 'SVM',
          mode = 'lines+markers'
      )
      dados_rf = go.Scatter(
          # Eixo x recebe a iteração
          x = iteracoes,
          # Eixo y recebe os valores de acurácia
          y = acc_rf,
          # Define o nome
          name = 'RandomForest',
          mode = 'lines+markers'
      )
      dados_nnet = go.Scatter(
          # Eixo x recebe a iteração
          x = iteracoes,
          # Eixo y recebe os valores de acurácia
          y = acc_nnet,
          # Define o nome
          name = 'Neural Net',
          mode = 'lines+markers'
      )

      # Alterando configurações de Layout do Gráfico
      layout = go.Layout(
          # Define Título
          title = 'Acurácia dos Classificadores conforme as Iterações',
          # Define o nome do eixo X
          xaxis = {'title': 'Iterações'},
          # Define o nome do eixo Y
          yaxis = {'title':'Acurácia'},
          # Define a cor da borda e contorno do gráfico
          paper_bgcolor='rgba(245, 246, 249, 1)',
          # Define a cor do fundo do gráfico
          plot_bgcolor='rgba(245, 246, 249, 1)'
      )

      # Plotando
      data = [dados_nb, dados_lr, dados_dt, dados_knn, dados_lda, dados_svm, dados_rf, dados_nnet]
      fig = go.Figure(data=data, layout=layout)
      # py.iplot(fig)
      pio.write_image(fig, 'images_active/'+str(selectedNormalization)+name[0]+name[1]+'_b_'+str(num_c)+'.png')
      # Chamando a função do gráfico interativo
      # configure_plotly_browser_state()

      # Precision = Daqueles que classifiquei como corretos, quantos efetivamente eram? (TP / (TP + FP))
      # Recall = Quando realmente é da classe X, o quão frequente você classifica como X?
      # F1-Score = Combina precisão e recall de modo a trazer um número único que indique a qualidade geral do modelo

      # Criando valores do eixo X
      eixo_x = ['GaussianNB', 'Logistic Regression', 'Decision Tree', 'k-NN', 'LDA', 'SVM', 'RandomForest', 'Neural Net']

      # Plotando o gráfico
      dados_precision = go.Bar(
          # Eixo x recebe o nome dos classificadores
          x = eixo_x,
          # Eixo y recebe os valores de acurácia
          y = precision,
          # Define o nome
          name = 'Precision',
      )

      # Plotando o gráfico
      dados_recall = go.Bar(
          # Eixo x recebe o nome dos classificadores
          x = eixo_x,
          # Eixo y recebe os valores de acurácia
          y = recall,
          # Define o nome
          name = 'Recall',
      )

      # Plotando o gráfico
      dados_f1score = go.Bar(
          # Eixo x recebe o nome dos classificadores
          x = eixo_x,
          # Eixo y recebe os valores de acurácia
          y = f1score,
          # Define o nome
          name = 'F1-Score',
      )

      # Alterando configurações de Layout do Gráfico
      layout = go.Layout(
          # Define Título
          title = 'Métricas de Avaliação',
          # Define o nome do eixo X
          xaxis = {'title': 'Classificadores'},
          # Define a cor da borda e contorno do gráfico
          paper_bgcolor='rgba(245, 246, 249, 1)',
          # Define a cor do fundo do gráfico
          plot_bgcolor='rgba(245, 246, 249, 1)'
      )

      # Plotando
      data = [dados_precision, dados_recall, dados_f1score]
      fig = go.Figure(data=data, layout=layout)
      # py.iplot(fig)
      pio.write_image(fig, 'images_active/'+str(selectedNormalization)+name[0]+name[1]+'_p_'+str(num_c)+'.png')

writeJSONFile(datastore)