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
import time
import warnings
warnings.filterwarnings("ignore")

# py.init_notebook_mode(connected=False)
def writeJSONFile(data):
  filePathNameWExt = 'results.json'
  with open(filePathNameWExt, 'a') as fp:
    json.dump(data, fp)

def configure_plotly_browser_state():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-1.43.1.min.js?noext',
            },
          });
        </script>
        '''))

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
resultsTxt = open("results.txt", "a")
resultsJson = open("results.json", "a")
datastore = []
for selectedNormalization in range(5):

  for item in listFile:
    raw_data = loadarff(item)

    df = pd.DataFrame(raw_data[0])

    df.head()

    X = df.iloc[:, 0:-1].values

    y = df['class']

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

    y = y_aux

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=327)

    #print('Tamanho do conjunto de Treino: {}'.format(X_train.shape))
    #print('Tamanho do conjunto de Teste: {}'.format(X_test.shape))
    
    name = item.split('/')
    descritor = name[1].split('.')[0]
    #resultsTxt.write(name[0])
    #resultsTxt.write('\n')

   # selectedNormalization = 2
    #resultsTxt.write('Normalization: '+str(selectedNormalization))
    #resultsTxt.write('\n')
    

    if selectedNormalization != 0:
      if selectedNormalization == 1:
        scaler = preprocessing.MinMaxScaler()
      if selectedNormalization == 2:
        scaler = preprocessing.StandardScaler()
      if selectedNormalization == 3:
        scaler = preprocessing.MaxAbsScaler()
      if selectedNormalization == 4:
        scaler = preprocessing.RobustScaler()

      X_train = scaler.fit_transform(X_train)
      X_test = scaler.transform(X_test)

    

    print('Media do Conjunto de Treinamento por Feature:')
    #print(X_train.mean(axis = 0))
    print('Desvio Padrao do Conjunto de Treinamento por Feature:')
    #print(X_train.std(axis = 0))


    t = time.time()
    gnb = GaussianNB()
    model1 = gnb.fit(X_train, y_train)
    #print('Treino do Gaussian Naive Bayes Terminado. (Tempo de execucao: {})'.format(time.time() - t))


    t = time.time()
    logreg = LogisticRegression()
    model2 = logreg.fit(X_train, y_train)
    #print('Treino do Logistic Regression Terminado. (Tempo de execucao: {})'.format(time.time() - t))


    t = time.time()
    dectree = DecisionTreeClassifier()
    model3 = dectree.fit(X_train, y_train)
    #print('Treino do Decision Tree Terminado. (Tempo de execucao: {})'.format(time.time() - t))


    t = time.time()
    knn = KNeighborsClassifier(n_neighbors = 3)
    model4 = knn.fit(X_train, y_train)
    #print('Treino do K-Nearest Neighbors Terminado. (Tempo de execucao: {})'.format(time.time() - t))
      

    t = time.time()
    lda = LinearDiscriminantAnalysis()
    model5 = lda.fit(X_train, y_train)
    #print('Treino do Linear Discriminant Analysis Terminado. (Tempo de execucao: {})'.format(time.time() - t))


    t = time.time()
    svm = SVC()
    model6 = svm.fit(X_train, y_train)
    #print('Treino do Support Vector Machine Terminado. (Tempo de execucao: {})'.format(time.time() - t))


    t = time.time()
    rf = RandomForestClassifier()
    model7 = rf.fit(X_train, y_train)
    #print('Treino do RandomForest Terminado. (Tempo de execucao: {})'.format(time.time() - t))
    

    t = time.time()
    nnet = MLPClassifier(alpha=1)
    model8 = nnet.fit(X_train, y_train)
    #print('Treino do Neural Net Terminado. (Tempo de execucao: {})'.format(time.time() - t))


    acc_train = []
    acc_test = []
    f1score = []
    precision = []
    recall = []


    t = time.time()

    aux = gnb.predict(X_test)

    cm = confusion_matrix(y_test, aux)

    f1score.append(f1_score(y_test, aux, average = 'macro'))

    precision.append(precision_score(y_test, aux, average = 'macro'))

    recall.append(recall_score(y_test, aux, average = 'macro'))

    acc_train.append(gnb.score(X_train, y_train))
    acc_test.append(gnb.score(X_test, y_test))

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
      'runningtime': '{:.5f}'.format(time.time() - t)
    }
    datastore.append(data)

    t = time.time()

    aux = logreg.predict(X_test)

    cm = confusion_matrix(y_test, aux)

    f1score.append(f1_score(y_test, aux, average = 'macro'))

    precision.append(precision_score(y_test, aux, average = 'macro'))

    recall.append(recall_score(y_test, aux, average = 'macro'))

    acc_train.append(logreg.score(X_train, y_train))
    acc_test.append(logreg.score(X_test, y_test))

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
      'runningtime': '{:.5f}'.format(time.time() - t)
    }
    datastore.append(data)


    t = time.time()

    aux = dectree.predict(X_test)

    cm = confusion_matrix(y_test, aux)

    f1score.append(f1_score(y_test, aux, average = 'macro'))

    precision.append(precision_score(y_test, aux, average = 'macro'))

    recall.append(recall_score(y_test, aux, average = 'macro'))

    acc_train.append(dectree.score(X_train, y_train))
    acc_test.append(dectree.score(X_test, y_test))

    data = {
      'normalization': selectedNormalization,
      'date': name[0],
      'extrator': descritor,
      'classifier':'tree',
      'treinamento': '{:.2f}'.format(acc_train[2]), 
      'teste': '{:.2f}'.format(acc_test[2]),
      'precision': '{:.5f}'.format(precision[2]),
      'recall': '{:.5f}'.format(recall[2]),
      'f1_score': '{:.5f}'.format(f1score[2]),
      'runningtime': '{:.5f}'.format(time.time() - t)
    }
    datastore.append(data)

    t = time.time()

    aux = knn.predict(X_test)

    cm = confusion_matrix(y_test, aux)

    f1score.append(f1_score(y_test, aux, average = 'macro'))

    precision.append(precision_score(y_test, aux, average = 'macro'))

    recall.append(recall_score(y_test, aux, average = 'macro'))

    acc_train.append(knn.score(X_train, y_train))
    acc_test.append(knn.score(X_test, y_test))

    data = {
      'normalization': selectedNormalization,
      'date': name[0],
      'extrator': descritor,
      'classifier':'neighbors',
      'treinamento': '{:.2f}'.format(acc_train[3]), 
      'teste': '{:.2f}'.format(acc_test[3]),
      'precision': '{:.5f}'.format(precision[3]),
      'recall': '{:.5f}'.format(recall[3]),
      'f1_score': '{:.5f}'.format(f1score[3]),
      'runningtime': '{:.5f}'.format(time.time() - t)
    }
    datastore.append(data)


    t = time.time()

    aux = lda.predict(X_test)

    cm = confusion_matrix(y_test, aux)

    f1score.append(f1_score(y_test, aux, average = 'macro'))

    precision.append(precision_score(y_test, aux, average = 'macro'))

    recall.append(recall_score(y_test, aux, average = 'macro'))

    acc_train.append(lda.score(X_train, y_train))
    acc_test.append(lda.score(X_test, y_test))

    data = {
      'normalization': selectedNormalization,
      'date': name[0],
      'extrator': descritor,
      'classifier':'linear',
      'treinamento': '{:.2f}'.format(acc_train[4]), 
      'teste': '{:.2f}'.format(acc_test[4]),
      'precision': '{:.5f}'.format(precision[4]),
      'recall': '{:.5f}'.format(recall[4]),
      'f1_score': '{:.5f}'.format(f1score[4]),
      'runningtime': '{:.5f}'.format(time.time() - t)
    }
    datastore.append(data)


    t = time.time()

    aux = svm.predict(X_test)

    cm = confusion_matrix(y_test, aux)

    f1score.append(f1_score(y_test, aux, average = 'macro'))

    precision.append(precision_score(y_test, aux, average = 'macro'))

    recall.append(recall_score(y_test, aux, average = 'macro'))

    acc_train.append(svm.score(X_train, y_train))
    acc_test.append(svm.score(X_test, y_test))

    data = {
      'normalization': selectedNormalization,
      'date': name[0],
      'extrator': descritor,
      'classifier':'support',
      'treinamento': '{:.2f}'.format(acc_train[5]), 
      'teste': '{:.2f}'.format(acc_test[5]),
      'precision': '{:.5f}'.format(precision[5]),
      'recall': '{:.5f}'.format(recall[5]),
      'f1_score': '{:.5f}'.format(f1score[5]),
      'runningtime': '{:.5f}'.format(time.time() - t)
    }
    datastore.append(data)


    t = time.time()

    aux = rf.predict(X_test)

    cm = confusion_matrix(y_test, aux)

    f1score.append(f1_score(y_test, aux, average = 'macro'))

    precision.append(precision_score(y_test, aux, average = 'macro'))

    recall.append(recall_score(y_test, aux, average = 'macro'))

    acc_train.append(rf.score(X_train, y_train))
    acc_test.append(rf.score(X_test, y_test))

    data = {
      'normalization': selectedNormalization,
      'date': name[0],
      'extrator': descritor,
      'classifier':'randomforest',
      'treinamento': '{:.2f}'.format(acc_train[6]), 
      'teste': '{:.2f}'.format(acc_test[6]),
      'precision': '{:.5f}'.format(precision[6]),
      'recall': '{:.5f}'.format(recall[6]),
      'f1_score': '{:.5f}'.format(f1score[6]),
      'runningtime': '{:.5f}'.format(time.time() - t)
    }
    datastore.append(data)


    t = time.time()

    aux = nnet.predict(X_test)

    cm = confusion_matrix(y_test, aux)

    f1score.append(f1_score(y_test, aux, average = 'macro'))

    precision.append(precision_score(y_test, aux, average = 'macro'))

    recall.append(recall_score(y_test, aux, average = 'macro'))

    acc_train.append(nnet.score(X_train, y_train))
    acc_test.append(nnet.score(X_test, y_test))

    data = {
      'normalization': selectedNormalization,
      'date': name[0],
      'extrator': descritor,
      'classifier':'neural',
      'treinamento': '{:.2f}'.format(acc_train[7]), 
      'teste': '{:.2f}'.format(acc_test[7]),
      'precision': '{:.5f}'.format(precision[7]),
      'recall': '{:.5f}'.format(recall[7]),
      'f1_score': '{:.5f}'.format(f1score[7]),
      'runningtime': '{:.5f}'.format(time.time() - t)
    }
    datastore.append(data)


    eixo_x = ['GaussianNB', 'Logistic Regression', 'Decision Tree', 'k-NN', 'LDA', 'SVM', 'RandomForest', 'Neural Net']


    dados_train = go.Bar(
        
        x = eixo_x,

        y = acc_train,
      
        name = 'Conjunto de Treino',
    )


    dados_test = go.Bar(

        x = eixo_x,

        y = acc_test,

        name = 'Conjunto de Teste',
    )


    layout = go.Layout(
    
        title = 'Acurácia dos Classificadores',

        xaxis = {'title': 'Classificadores'},

        yaxis = {'title':'Acurácia'},

        paper_bgcolor='rgba(245, 246, 249, 1)',

        plot_bgcolor='rgba(245, 246, 249, 1)'
    )


    data1 = [dados_train, dados_test]
    fig = go.Figure(data=data1, layout=layout)
    # py.iplot(fig)


    # configure_plotly_browser_state()




    eixo_x = ['GaussianNB', 'Logistic Regression', 'Decision Tree', 'k-NN', 'LDA', 'SVM', 'RandomForest', 'Neural Net']


    dados_precision = go.Scatter(

        x = eixo_x,

        y = precision,

        name = 'Precision',
        mode = 'lines+markers'
    )


    dados_recall = go.Scatter(

        x = eixo_x,

        y = recall,

        name = 'Recall',
        mode = 'lines+markers'
    )


    dados_f1score = go.Scatter(

        x = eixo_x,

        y = f1score,

        name = 'F1-Score',
        mode = 'lines+markers'
    )


    layout = go.Layout(

        title = 'Métricas de Avaliação',

        xaxis = {'title': 'Classificadores'},

        paper_bgcolor='rgba(245, 246, 249, 1)',

        plot_bgcolor='rgba(245, 246, 249, 1)'
    )


    #data1 = [dados_precision, dados_recall, dados_f1score]
    #fig = go.Figure(data=data1, layout=layout)
    # py.plot(fig)
    #pio.write_image(fig, 'images/'+name[1]+'.png')

writeJSONFile(datastore)