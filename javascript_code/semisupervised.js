var express = require('express')
var fs = require('fs')
var json = fs.readFileSync('../results_semi.json', 'utf8')

var app = express();
var results = JSON.parse(json)
app.use(express.static(__dirname + '/public'))

app.get('/', function (req, res) {
  res.render('index')
})
app.get('/min-time', function (req, res) {
  var runningtimes = results.map((item) => +item.runningtime)
  var min_runningtimes = Math.min.apply(null, runningtimes)
  var value = results.filter((item) => Math.abs(item.runningtime - min_runningtimes) < Number.EPSILON)
  
  // console.log('menor tempo de execução')
  // console.log(value)
  res.json(value)
})
app.get('/max-time', function (req, res) {
  var runningtimes = results.map((item) => +item.runningtime)
  var max_runningtimes = Math.max.apply(null, runningtimes)
  var value = results.filter((item) => Math.abs(item.runningtime - max_runningtimes) < Number.EPSILON)

  // console.log('menor tempo de execução')
  // console.log(value)
  res.json(value)
})
app.get('/recall', function (req, res) {
  var recalls = results.map((item) => +item.recall)
  var max_recall = Math.max.apply(null, recalls)
  var value = results.filter((item) => Math.abs(item.recall - max_recall) < Number.EPSILON)

  // console.log(value)
  res.json(value)
})
app.get('/test', function (req, res) {

  var tests = results.map((item) => +item.teste)
  var max_tests = Math.max.apply(null, tests)
  var value = results.filter((item) => Math.abs(item.teste - max_tests) < Number.EPSILON)
  
  // console.log('maior acuracia de teste')
  // console.log(value)
  res.json(value)
})
app.get('/trainning', function (req, res) {

  var trainnings = results.map((item) => +item.treinamento)
  var max_trainning = Math.max.apply(null, trainnings)
  var value = results.filter((item) => Math.abs(item.treinamento - max_trainning) < Number.EPSILON)
  // console.log('maior acuracia de treinamento')
  // console.log(value)
  res.json(value)
})
app.get('/fscore', function (req, res) {

  var fscores = results.map((item) => +item.f1_score)
  max_fscore = Math.max.apply(null, fscores)
  var value = results.filter((item) => Math.abs(item.f1_score - max_fscore) < Number.EPSILON)
  // console.log('maior fscore')
  // console.log(value)
  res.json(value)
})
app.get('/precision', function (req, res) {

  var precisions = results.map((item) => +item.precision)
  max_precision = Math.max.apply(null, precisions)
  var value = results.filter((item) => Math.abs(item.precision - max_precision) < Number.EPSILON)
  //console.log('maior precisao')
  //console.log(value)
  res.json(value)
})
app.get('/iai', function (req, res) {
  var x = results.filter((item) => item.normalization == 1)
  res.json(x)
})
app.listen(3000, function () {
  console.log('Listening in port 3000')
})