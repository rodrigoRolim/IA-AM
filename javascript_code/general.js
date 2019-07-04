var express = require('express')
var fs = require('fs')
var json_1 = fs.readFileSync('../results_active.json', 'utf8')
var json_2 = fs.readFileSync('../results.json', 'utf8')
var json_3 = fs.readFileSync('../results_semi.json', 'utf8')
var json = [...JSON.parse(json_1), ...JSON.parse(json_2), ...JSON.parse(json_3)]

var app = express()
app.use(express.static(__dirname + '/public'))
var Performance = require('./functions/performance')

var results = json

var performance = new Performance(results)

app.get('/total', function (req, res) {
  res.json(results)
})
app.get('/', function (req, res) {
  res.render('index')
})
app.get('/min-time', function (req, res) {
  res.json(performance.min_time())
})
app.get('/max-time', function (req, res) {
  res.json(performance.max_time())
})
app.get('/recall', function (req, res) {
  res.json(performance.recall())
})
app.get('/test', function (req, res) {
  res.json(performance.test_accuracy())
})
app.get('/trainning', function (req, res) {
  res.json(performance.trainning_accuracy())
})
app.get('/fscore', function (req, res) {
  res.json(performance.fscore())
})
app.get('/precision', function (req, res) {
  res.json(performance.max_precision())
})
app.get('/oftener', function (req, res) {
  res.json(performance.frenquency())
})
app.get('/number', function (req, res) {
  res.json(performance.numIterations())
})
app.listen(3004, function () {
  console.log('Listening on the port 3004')
})