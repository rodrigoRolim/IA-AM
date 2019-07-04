function Performance (results) {
  this._results = results
}
Performance.prototype.max_precision = function () {

  var precisions = this._results.map((item) => +item.precision)
  max_precision = Math.max.apply(null, precisions)
  var value = this._results.filter((item) => Math.abs(item.precision - max_precision) < Number.EPSILON)

  return value
}
Performance.prototype.min_time = function () {

  var runningtimes = this._results.map((item) => +item.runningtime)
  var min_runningtimes = Math.min.apply(null, runningtimes)
  var value = this._results.filter((item) => Math.abs(item.runningtime - min_runningtimes) < Number.EPSILON)

  return value
}
Performance.prototype.max_time = function () {

  var runningtimes = this._results.map((item) => +item.runningtime)
  var max_runningtimes = Math.max.apply(null, runningtimes)
  var value = this._results.filter((item) => Math.abs(item.runningtime - max_runningtimes) < Number.EPSILON)

  return value
}
Performance.prototype.recall = function () {

  var recalls = this._results.map((item) => +item.recall)
  var max_recall = Math.max.apply(null, recalls)
  var value = this._results.filter((item) => Math.abs(item.recall - max_recall) < Number.EPSILON)
  
  return value
}
Performance.prototype.test_accuracy = function () {

  var tests = this._results.map((item) => +item.teste)
  var max_tests = Math.max.apply(null, tests)
  var value = this._results.filter((item) => Math.abs(item.teste - max_tests) < Number.EPSILON)

  return value
}
Performance.prototype.trainning_accuracy = function () {
  
  var trainnings = this._results.map((item) => +item.treinamento)
  var max_trainning = Math.max.apply(null, trainnings)
  var value = this._results.filter((item) => Math.abs(item.treinamento - max_trainning) < Number.EPSILON)

  return value
}
Performance.prototype.fscore = function () {
  
  var fscores = this._results.map((item) => +item.f1_score)
  max_fscore = Math.max.apply(null, fscores)
  var value = this._results.filter((item) => Math.abs(item.f1_score - max_fscore) < Number.EPSILON)

  return value
}
Performance.prototype.frenquency = function () {
  var runningtimes = this._results.map((item) => +item.runningtime)
  var min_runningtimes = Math.min.apply(null, runningtimes)
  var res_run = this._results.filter((item) => Math.abs(item.runningtime - min_runningtimes) < Number.EPSILON)
  
  var recalls = this._results.map((item) => +item.recall)
  var max_recall = Math.max.apply(null, recalls)
  var res_recall = this._results.filter((item) => Math.abs(item.recall - max_recall) < Number.EPSILON)
  
  var tests = this._results.map((item) => +item.teste)
  var max_tests = Math.max.apply(null, tests)
  var res_tests = this._results.filter((item) => Math.abs(item.teste - max_tests) < Number.EPSILON)

  var trainnings = this._results.map((item) => +item.treinamento)
  var max_trainning = Math.max.apply(null, trainnings)
  var res_trainning = this._results.filter((item) => Math.abs(item.treinamento - max_trainning) < Number.EPSILON)

  var fscores = this._results.map((item) => +item.f1_score)
  max_fscore = Math.max.apply(null, fscores)
  var res_fscore = this._results.filter((item) => Math.abs(item.f1_score - max_fscore) < Number.EPSILON)

  var precisions = this._results.map((item) => +item.precision)
  max_precision = Math.max.apply(null, precisions)
  var res_precision = this._results.filter((item) => Math.abs(item.precision - max_precision) < Number.EPSILON)

  let maxs = []
  let f_extractor = {}
  let f_classifier = {}
  let f_normalization = {}
  let f_date = {}


  maxs.push(res_fscore)
  maxs.push(res_precision)
  maxs.push(res_recall)
  maxs.push(res_run)
  maxs.push(res_tests)
  maxs.push(res_trainning)

  maxs.forEach((item) => {

    if (f_extractor.hasOwnProperty(item[0].extrator)) {
      f_extractor[item[0].extrator]++
    } else {
      f_extractor[item[0].extrator] = 1
    }

    if (f_classifier.hasOwnProperty(item[0].classifier)) {
      f_classifier[item[0].classifier]++
    } else {
      f_classifier[item[0].classifier] = 1
    }

    if (f_normalization.hasOwnProperty(item[0].normalization)) {
      f_normalization[item[0].normalization]++
    } else {
      f_normalization[item[0].normalization] = 1
    }

    if (f_date.hasOwnProperty(item[0].date)) {
      f_date[item[0].date]++
    } else {
      f_date[item[0].date] = 1
    }
  })
  
  var values = []
  values.push(f_classifier)
  values.push(f_date)
  values.push(f_extractor)
  values.push(f_normalization)

  return values
}
Performance.prototype.all = function () {
  return this._results;
}
module.exports = Performance