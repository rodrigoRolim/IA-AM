var express = require('express')
var fs = require('fs')
var json = fs.readFileSync('../results.json', 'utf8')
var app = express()

var Performance = require('./functions/performance')


app.use(express.static(__dirname + '/public'))
var results = JSON.parse(json)
var performance = new Performance(results)

app.get('/', function (req, res) {
  res.render('index')
})
app.get('/min-time', function (req, res) {
  res.json(performance.max_time())
})
app.get('/max-time', function (req, res) {
  res.json(performance.min_time())
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

app.listen(3000, function () {
  console.log('Listening in port 3000')
})
