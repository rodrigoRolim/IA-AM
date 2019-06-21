var fs = require('fs')
var json = fs.readFileSync('../results.json', 'utf8')

var results = JSON.parse(json)

// maior precisão
/* var precisions = results.map(function(item){
  return +item.precision
})

max_precision = Math.max.apply(null, precisions)

var value = results.filter((item) => Math.abs(item.precision - max_precision) < Number.EPSILON)
console.log('maior precisao')
console.log(value) */
// =========

// maior fscore: 2/1/recall + 1/precision

/* var fscores = results.map(function(item) {
  return +item.f1_score
})

max_fscore = Math.max.apply(null, fscores)

var value = results.filter((item) => Math.abs(item.f1_score - max_fscore) < Number.EPSILON)
console.log('maior fscore')
console.log(value)  */

// maior acuracia de treinamento

/* var trainnings = results.map(function(item) {
  return +item.treinamento
})

var max_trainning = Math.max.apply(null, trainnings)

var value = results.filter((item) => Math.abs(item.treinamento - max_trainning) < Number.EPSILON)
console.log('maior acuracia de treinamento')
console.log(value) */

// maior acuracia de teste

/*  var tests = results.map(function(item) {
  return +item.teste
})

var max_tests = Math.max.apply(null, tests)

var value = results.filter((item) => Math.abs(item.teste - max_tests) < Number.EPSILON)

console.log('maior acuracia de teste')
console.log(value) */

// menor tempo de execução

/* var runningtimes = results.map((item) => +item.runningtime)
var min_runningtimes = Math.min.apply(null, runningtimes)
var value = results.filter((item) => Math.abs(item.runningtime - min_runningtimes) < Number.EPSILON)

console.log('menor tempo de execução')
console.log(value) */

var runningtimes = results.map((item) => +item.runningtime)
var max_runningtimes = Math.max.apply(null, runningtimes)
var value = results.filter((item) => Math.abs(item.runningtime - max_runningtimes) < Number.EPSILON)

console.log('menor tempo de execução')
console.log(value)