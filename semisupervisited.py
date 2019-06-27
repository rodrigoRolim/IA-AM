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
from sklearn.semi_supervised import LabelPropagation
import time
import warnings
warnings.filterwarnings("ignore")

# py.init_notebook_mode(connected=False)

def writeJSONFile(data):
  filePathNameWExt = 'results_semi.json'
  with open(filePathNameWExt, 'a') as fp:
    json.dump(data, fp)


listFile = ['alinevspredator/AutoColorCorrelogram.arff',
            'alinevspredator/FCTH.arff',
            'alinevspredator/JCD.arff',
            'alinevspredator/LBP.arff',
            'alinevspredator/Moments.arff',
            'alinevspredator/MPO.arff',
            'alinevspredator/MPOC.arff',
            'alinevspredator/PHOG.arff',
            'scissors/AutoColorCorrelogram.arff',
            'scissors/FCTH.arff',
            'scissors/JCD.arff',
            'scissors/LBP.arff',
            'scissors/Moments.arff',
            'scissors/MPO.arff',
            'scissors/MPOC.arff',
            'scissors/PHOG.arff',
            'shapes/AutoColorCorrelogram.arff',
            'shapes/FCTH.arff',
            'shapes/JCD.arff',
            'shapes/LBP.arff',
            'shapes/Moments.arff',
            'shapes/MPO.arff',
            'shapes/MPOC.arff',
            'shapes/PHOG.arff',
            'vegetables/AutoColorCorrelogram.arff',
            'vegetables/LBP.arff',
            'vegetables/FCTH.arff',
            'vegetables/JCD.arff',
            'vegetables/LBP.arff',
            'vegetables/Moments.arff',
            'vegetables/MPO.arff',
            'vegetables/MPOC.arff',
            'vegetables/PHOG.arff'
            ]
datastore = [] 
for selectedNormalization in range(5):
  
  for item in listFile:
    for l in range(1, 5):
      qtd_test = round(l/10, 1)
      qtd_trainning = round(1 - qtd_test, 1)
      # Carrega o .arff
      raw_data = loadarff(item)
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
      X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=qtd_test,random_state=327)

      # print('Tamanho do conjunto de Treino: {}'.format(X_train.shape))
      # print('Tamanho do conjunto de Teste: {}'.format(X_test.shape))
      name = item.split('/')
      descritor = name[1].split('.')[0]
      # Escolha umas das 4 técnicas de normalização existentes
      # 1 = MinMaxScaler, 2 = StandardScaler, 3 = MaxAbsScaler, 4 = RobustScaler
      selectedNormalization = 1

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

      # Dividindo o conjunto em 20% rotuladas e 80% não rotulada.
      # O parâmetro random_state = 327 define que sempre será dividido da mesma forma o conjunto.
      X_train_labeled, X_test_unlabeled, y_train_labeled, y_test_unlabeled = train_test_split(X_train, y_train, test_size=qtd_trainning,random_state=327)

      print('Tamanho do conjunto Rotulado: {}'.format(X_train_labeled.shape))
      print('Tamanho do conjunto Não Rotulado: {}'.format(X_test_unlabeled.shape))

      # Transforma a lista em Numpy Array
      y_train_labeled = np.asarray(y_train_labeled)
      # Empilhando os rótulos
      y_train_labeled = np.vstack(y_train_labeled)
      # Obtem o comprimento dos rotulos da parte não rotulada
      length = len(y_test_unlabeled)
      # Pelo método, só serão propagado valores, para os rótulos que tiverem valor -1 por isso:
      # Cria um array com esse tamanho de valores -1
      unlabeled_set = np.arange(length)
      for i in unlabeled_set:
        unlabeled_set[i] = -1
      # Empilha os valores de -1
      y_test_unlabeled = np.vstack(unlabeled_set)

      # Concatenando os rótulos
      labels = np.concatenate((y_train_labeled, y_test_unlabeled), axis = 0)
      # Concatena os dados com sua parte não rotulada do conjunto
      X_train = np.concatenate((X_train_labeled, X_test_unlabeled), axis = 0)

      # Propagando os rótulos com Label Propagation
      lp_model = LabelPropagation(gamma=15, max_iter=30, kernel = 'rbf')
      lp_model.fit(X_train, labels)
      print('')
      print(lp_model)
      y_train = lp_model.transduction_

      # Inicializar os classificadores

      # Gaussian Naive Bayes
      t = time.time()
      gnb = GaussianNB()
      model1 = gnb.fit(X_train, y_train)
      print('Treino do Gaussian Naive Bayes Terminado. (Tempo de execucao: {})'.format(time.time() - t))

      # Logistic Regression
      t = time.time()
      logreg = LogisticRegression()
      model2 = logreg.fit(X_train, y_train)
      print('Treino do Logistic Regression Terminado. (Tempo de execucao: {})'.format(time.time() - t))

      # Decision Tree
      t = time.time()
      dectree = DecisionTreeClassifier()
      model3 = dectree.fit(X_train, y_train)
      print('Treino do Decision Tree Terminado. (Tempo de execucao: {})'.format(time.time() - t))

      # K-Nearest Neighbors
      t = time.time()
      knn = KNeighborsClassifier(n_neighbors = 3)
      model4 = knn.fit(X_train, y_train)
      print('Treino do K-Nearest Neighbors Terminado. (Tempo de execucao: {})'.format(time.time() - t))
        
      # Linear Discriminant Analysis
      t = time.time()
      lda = LinearDiscriminantAnalysis()
      model5 = lda.fit(X_train, y_train)
      print('Treino do Linear Discriminant Analysis Terminado. (Tempo de execucao: {})'.format(time.time() - t))

      # Support Vector Machine
      t = time.time()
      svm = SVC()
      model6 = svm.fit(X_train, y_train)
      print('Treino do Support Vector Machine Terminado. (Tempo de execucao: {})'.format(time.time() - t))

      # RandomForest
      t = time.time()
      rf = RandomForestClassifier()
      model7 = rf.fit(X_train, y_train)
      print('Treino do RandomForest Terminado. (Tempo de execucao: {})'.format(time.time() - t))

      # Neural Net
      t = time.time()
      nnet = MLPClassifier(alpha=1)
      model8 = nnet.fit(X_train, y_train)
      print('Treino do Neural Net Terminado. (Tempo de execucao: {})'.format(time.time() - t))

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
      # print('Acuracia obtida com o Gaussian Naive Bayes no Conjunto de Treinamento: {:.2f}'.format(acc_train[0]))
      # print('Acuracia obtida com o Gaussian Naive Bayes no Conjunto de Teste: {:.2f}'.format(acc_test[0]))
      # print('Matriz de Confusão:')
      # print(cm)
      # print('Precision: {:.5f}'.format(precision[0]))
      # print('Recall: {:.5f}'.format(recall[0]))
      # print('F1-score: {:.5f}'.format(f1score[0]))
      # print('(Tempo de execucao: {:.5f})'.format(time.time() - t))
      # print('')
      data = {
        'normalization': selectedNormalization,
        'date': name[0],
        'extrator': descritor,
        'classifier': 'gaussian',
        'treinamento': '{:.2f}'.format(acc_train[0]), 
        'teste': '{:.2f}'.format(acc_test[0]),
        'precision': '{:.5f}'.format(precision[0]),
        'recall': '{:.5f}'.format(recall[0]),
        'f1_score': '{:.5f}'.format(f1score[0]),
        'runningtime': '{:.5f}'.format(time.time() - t),
        'labels': qtd_test,
        'notlabels': qtd_trainning
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
      # print('Acuracia obtida com o Logistic Regression no Conjunto de Treinamento: {:.2f}'.format(acc_train[1]))
      # print('Acuracia obtida com o Logistic Regression no Conjunto de Teste: {:.2f}'.format(acc_test[1]))
      # print('Matriz de Confusão:')
      # print(cm)
      # print('Precision: {:.5f}'.format(precision[1]))
      # print('Recall: {:.5f}'.format(recall[1]))
      # print('F1-score: {:.5f}'.format(f1score[1]))
      # print('(Tempo de execucao: {:.5f})'.format(time.time() - t))
      # print('')
      data = {
        'normalization': selectedNormalization,
        'date': name[0],
        'extrator': descritor,
        'classifier':'regression',
        'treinamento': '{:.2f}'.format(acc_train[1]), 
        'teste': '{:.2f}'.format(acc_test[1]),
        'precision': '{:.5f}'.format(precision[1]),
        'recall': '{:.5f}'.format(recall[1]),
        'f1_score': '{:.5f}'.format(f1score[1]),
        'runningtime': '{:.5f}'.format(time.time() - t),
        'labels': qtd_test,
        'notlabels': qtd_trainning
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
      # print('Acuracia obtida com o Decision Tree no Conjunto de Treinamento: {:.2f}'.format(acc_train[2]))
      # print('Acuracia obtida com o Decision Tree no Conjunto de Teste: {:.2f}'.format(acc_test[2]))
      # print('Matriz de Confusão:')
      # print(cm)
      # print('Precision: {:.5f}'.format(precision[2]))
      # print('Recall: {:.5f}'.format(recall[2]))
      # print('F1-score: {:.5f}'.format(f1score[2]))
      # print('(Tempo de execucao: {:.5f})'.format(time.time() - t))
      # print('')
      data = {
        'normalization': selectedNormalization,
        'date': name[0],
        'extrator': descritor,
        'classifier':'regression',
        'treinamento': '{:.2f}'.format(acc_train[2]), 
        'teste': '{:.2f}'.format(acc_test[2]),
        'precision': '{:.5f}'.format(precision[2]),
        'recall': '{:.5f}'.format(recall[2]),
        'f1_score': '{:.5f}'.format(f1score[2]),
        'runningtime': '{:.5f}'.format(time.time() - t),
        'labels': qtd_test,
        'notlabels': qtd_trainning
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
      # print('Acuracia obtida com o K-Nearest Neighbors no Conjunto de Treinamento: {:.2f}'.format(acc_train[3]))
      # print('Acuracia obtida com o K-Nearest Neighbors no Conjunto de Teste: {:.2f}'.format(acc_test[3]))
      # print('Matriz de Confusão:')
      # print(cm)
      # print('Precision: {:.5f}'.format(precision[3]))
      # print('Recall: {:.5f}'.format(recall[3]))
      # print('F1-score: {:.5f}'.format(f1score[3]))
      # print('(Tempo de execucao: {:.5f})'.format(time.time() - t))
      # print('')
      data = {
        'normalization': selectedNormalization,
        'date': name[0],
        'extrator': descritor,
        'classifier':'regression',
        'treinamento': '{:.2f}'.format(acc_train[3]), 
        'teste': '{:.2f}'.format(acc_test[3]),
        'precision': '{:.5f}'.format(precision[3]),
        'recall': '{:.5f}'.format(recall[3]),
        'f1_score': '{:.5f}'.format(f1score[3]),
        'runningtime': '{:.5f}'.format(time.time() - t),
        'labels': qtd_test,
        'notlabels': qtd_trainning
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
      # print('Acuracia obtida com o Linear Discriminant Analysis no Conjunto de Treinamento: {:.2f}'.format(acc_train[4]))
      # print('Acuracia obtida com o Linear Discriminant Analysis no Conjunto de Teste: {:.2f}'.format(acc_test[4]))
      # print('Matriz de Confusão:')
      # print(cm)
      # print('Precision: {:.5f}'.format(precision[4]))
      # print('Recall: {:.5f}'.format(recall[4]))
      # print('F1-score: {:.5f}'.format(f1score[4]))
      # print('(Tempo de execucao: {:.5f})'.format(time.time() - t))
      # print('')
      data = {
        'normalization': selectedNormalization,
        'date': name[0],
        'extrator': descritor,
        'classifier':'tree',
        'treinamento': '{:.2f}'.format(acc_train[4]), 
        'teste': '{:.2f}'.format(acc_test[4]),
        'precision': '{:.5f}'.format(precision[4]),
        'recall': '{:.5f}'.format(recall[4]),
        'f1_score': '{:.5f}'.format(f1score[4]),
        'runningtime': '{:.5f}'.format(time.time() - t),
        'labels': qtd_test,
        'notlabels': qtd_trainning
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
      # print('Acuracia obtida com o Support Vector Machine no Conjunto de Treinamento: {:.2f}'.format(acc_train[5]))
      # print('Acuracia obtida com o Support Vector Machine no Conjunto de Teste: {:.2f}'.format(acc_test[5]))
      # print('Matriz de Confusão:')
      # print(cm)
      # print('Precision: {:.5f}'.format(precision[5]))
      # print('Recall: {:.5f}'.format(recall[5]))
      # print('F1-score: {:.5f}'.format(f1score[5]))
      # print('(Tempo de execucao: {:.5f})'.format(time.time() - t))
      # print('')
      data = {
        'normalization': selectedNormalization,
        'date': name[0],
        'extrator': descritor,
        'classifier':'neighbors',
        'treinamento': '{:.2f}'.format(acc_train[5]), 
        'teste': '{:.2f}'.format(acc_test[5]),
        'precision': '{:.5f}'.format(precision[5]),
        'recall': '{:.5f}'.format(recall[5]),
        'f1_score': '{:.5f}'.format(f1score[5]),
        'runningtime': '{:.5f}'.format(time.time() - t),
        'labels': qtd_test,
        'notlabels': qtd_trainning
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
      # print('Acuracia obtida com o RandomForest no Conjunto de Treinamento: {:.2f}'.format(acc_train[6]))
      # print('Acuracia obtida com o RandomForest no Conjunto de Teste: {:.2f}'.format(acc_test[6]))
      # print('Matriz de Confusão:')
      # print(cm)
      # print('Precision: {:.5f}'.format(precision[6]))
      # print('Recall: {:.5f}'.format(recall[6]))
      # print('F1-score: {:.5f}'.format(f1score[6]))
      # print('(Tempo de execucao: {:.5f})'.format(time.time() - t))
      # print('')
      data = {
        'normalization': selectedNormalization,
        'date': name[0],
        'extrator': descritor,
        'classifier':'neighbors',
        'treinamento': '{:.2f}'.format(acc_train[6]), 
        'teste': '{:.2f}'.format(acc_test[6]),
        'precision': '{:.5f}'.format(precision[6]),
        'recall': '{:.5f}'.format(recall[6]),
        'f1_score': '{:.5f}'.format(f1score[6]),
        'runningtime': '{:.5f}'.format(time.time() - t),
        'labels': qtd_test,
        'notlabels': qtd_trainning
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
      # print('Acuracia obtida com o Neural Net no Conjunto de Treinamento: {:.2f}'.format(acc_train[7]))
      # print('Acuracia obtida com o Neural Net no Conjunto de Teste: {:.2f}'.format(acc_test[7]))
      # print('Matriz de Confusão:')
      # print(cm)
      # print('Precision: {:.5f}'.format(precision[7]))
      # print('Recall: {:.5f}'.format(recall[7]))
      # print('F1-score: {:.5f}'.format(f1score[7]))
      # print('(Tempo de execucao: {:.5f})'.format(time.time() - t))
      # print('')
      data = {
        'normalization': selectedNormalization,
        'date': name[0],
        'extrator': descritor,
        'classifier':'neighbors',
        'treinamento': '{:.2f}'.format(acc_train[7]), 
        'teste': '{:.2f}'.format(acc_test[7]),
        'precision': '{:.5f}'.format(precision[7]),
        'recall': '{:.5f}'.format(recall[7]),
        'f1_score': '{:.5f}'.format(f1score[7]),
        'runningtime': '{:.5f}'.format(time.time() - t),
        'labels': qtd_test,
        'notlabels': qtd_trainning
      }
      datastore.append(data)
      # Chamando a função do gráfico interativo
      # configure_plotly_browser_state()

      # Criando valores do eixo X
      eixo_x = ['GaussianNB', 'Logistic Regression', 'Decision Tree', 'k-NN', 'LDA', 'SVM', 'RandomForest', 'Neural Net']

      # Plotando o gráfico
      dados_train = go.Bar(
          # Eixo x recebe o nome dos classificadores
          x = eixo_x,
          # Eixo y recebe os valores de acurácia
          y = acc_train,
          # Define o nome
          name = 'Conjunto de Treino',
      )

      # Plotando o gráfico
      dados_test = go.Bar(
          # Eixo x recebe o nome dos classificadores
          x = eixo_x,
          # Eixo y recebe os valores de acurácia
          y = acc_test,
          # Define o nome
          name = 'Conjunto de Teste',
      )

      # Alterando configurações de Layout do Gráfico
      layout = go.Layout(
          # Define Título
          title = 'Acurácia dos Classificadores',
          # Define o nome do eixo X
          xaxis = {'title': 'Classificadores'},
          # Define o nome do eixo Y
          yaxis = {'title':'Acurácia'},
          # Define a cor da borda e contorno do gráfico
          paper_bgcolor='rgba(245, 246, 249, 1)',
          # Define a cor do fundo do gráfico
          plot_bgcolor='rgba(245, 246, 249, 1)'
      )

      # Plotando
      data = [dados_train, dados_test]
      fig = go.Figure(data=data, layout=layout)
      # py.iplot(fig)
      pio.write_image(fig, 'images_semi/'+name[0]+name[1]+'_b'+'.png')

      # Chamando a função do gráfico interativo
      # configure_plotly_browser_state()

      # Precision = Daqueles que classifiquei como corretos, quantos efetivamente eram? (TP / (TP + FP))
      # Recall = Quando realmente é da classe X, o quão frequente você classifica como X?
      # F1-Score = Combina precisão e recall de modo a trazer um número único que indique a qualidade geral do modelo

      # Criando valores do eixo X
      eixo_x = ['GaussianNB', 'Logistic Regression', 'Decision Tree', 'k-NN', 'LDA', 'SVM', 'RandomForest', 'Neural Net']

      # Plotando o gráfico
      dados_precision = go.Scatter(
          # Eixo x recebe o nome dos classificadores
          x = eixo_x,
          # Eixo y recebe os valores de acurácia
          y = precision,
          # Define o nome
          name = 'Precision',
          mode = 'lines+markers'
      )

      # Plotando o gráfico
      dados_recall = go.Scatter(
          # Eixo x recebe o nome dos classificadores
          x = eixo_x,
          # Eixo y recebe os valores de acurácia
          y = recall,
          # Define o nome
          name = 'Recall',
          mode = 'lines+markers'
      )

      # Plotando o gráfico
      dados_f1score = go.Scatter(
          # Eixo x recebe o nome dos classificadores
          x = eixo_x,
          # Eixo y recebe os valores de acurácia
          y = f1score,
          # Define o nome
          name = 'F1-Score',
          mode = 'lines+markers'
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
      pio.write_image(fig, 'images_semi/'+str(selectedNormalization)+name[0]+name[1]+'_p'+'.png')

writeJSONFile(datastore)