var express = require('express')
var fs = require('fs')
var json = fs.readFileSync('../results_no.json', 'utf8')

var app = express()
var results = JSON.parse(json)

app.get('/', function (req, res) {
  res.json(results)
})
app.listen(3002, function () {
  console.log('Listening on the port 3002')
})
