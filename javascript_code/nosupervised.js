var express = require('express')
var fs = require('fs')
var json = fs.readFileSync('../results_no.json', 'utf8')

var app = express()
var results = JSON.parse(json)
var Performance = require('./functions/performance')
var results = JSON.parse(json)
var performance = new Performance(results)
app.get('/', function (req, res) {
  res.json(results)
})
app.get('/best-k', function (req, res) {

  var best_k_supervised = results.filter((item) => item.dataset == 'scissors' && item.normalization == 2 && item.descritor == 'PHOG')
  var best_k_nosupervised = results.filter((item) => item.dataset == 'shapes' && item.normalization == 1 && item.descritor == 'PHOG')
  res.json([best_k_supervised, best_k_nosupervised])
})
app.get('/number', function (req, res) {
  res.json(performance.numIterations())
})
app.listen(3002, function () {
  console.log('Listening on the port 3002')
})
