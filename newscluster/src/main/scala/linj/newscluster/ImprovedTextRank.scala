package linj.newscluster

import linj.common.Blas

import scala.collection.mutable

/**
  * Created by linjiang on 2017/5/8.
  */
class RankConf extends Serializable {
  var test = false
  var batchTest = false
  var useExp = false
  var windowSize = 5
  var damp = 0.85
  var iter = 20
  var topN = 4
  var titleWeight = 10.0
  val minScoreDif = 0.0001

  var weighConf: Map[String, Int] = Map("u"->10)

  override def toString: String =
  s"useage : batch=$batchTest test=$test, iter=$iter, window=$windowSize, damp=$damp, titleWeight=$titleWeight" +
    weighConf.toString()
}

class ImprovedTextRanker(conf: RankConf,
                title: Array[String], content: Array[Array[String]],
                wordVec: Map[String, Array[Float]],
                wordIdf: Map[String, Float] = Map.empty.withDefaultValue(1)) {

  private val wordGraph = buildWordGraph()
  private val wordCnt = wordGraph.size
  private val wordPositionWeight: Map[String, Float] = {
    val ret = new mutable.HashMap[String, Float]()

    content.foreach(sen => {
      sen.foreach(w => {
        ret += w -> 1.0f
      })
    })

    title.foreach(w => {
      ret += w -> conf.titleWeight.toFloat
    })
    ret.toMap
  }
  private val wordOutPositionWeightSum: Map[String, Float] = {
    wordGraph.map(item => {
      val w = item._1
      val weighSum = item._2.map(e => wordPositionWeight(e)).sum
      (w, weighSum)
    })
  }

  private val wordOutIdfWeightSum: Map[String, Float] = {
    wordGraph.map(e => {
      val w = e._1
      val weighSum = e._2.map(e2 => wordIdf(e2)).sum
      (w, weighSum)
    })
  }

  private val simMap = new mutable.HashMap[(String, String), Double]()

  private def similarity(w1: String, w2: String): Double = {
    val key = if (w1 > w2) (w2, w1) else (w1, w2)
    if (!simMap.contains(key)) {
      var f = Blas.dot(wordVec(w1), wordVec(w2))
      simMap.put(key, f)
    }
    simMap(key)
  }

  private val wFunMap = Map(
    "s" -> simWeight _,
    "u" -> uniformWeight,
    "p" -> positionWeight _,
    "i" -> idfWeight _
  )

  private def simWeight(w1: String, w2: String): Double = {
    similarity(w1, w2) / wordGraph(w1).length //  |out(w1)|
  }

  private def uniformWeight = (w1: String, w2: String) => 1.0 / wordGraph(w1).length

  private def positionWeight(w1: String, w2: String): Double = {
    wordPositionWeight(w2) / wordOutPositionWeightSum(w1)
  }

  private def idfWeight(w1: String, w2: String): Double = {
    wordIdf(w2) / wordOutIdfWeightSum(w1)
  }

  private def buildWordGraph(): Map[String, Array[String]] = {

    val wordsLinks = new mutable.HashMap[String, mutable.Set[String]]().withDefaultValue(new mutable.HashSet[String]())

    title.foreach(w => wordsLinks += w -> (mutable.HashSet(title: _*) -= w))

    content.foreach(s => {
      s.sliding(conf.windowSize).foreach(window => {
        for (i <- 0 until window.length) {
          val word = window(i)
          val linkNodes = new mutable.HashSet[String]
          linkNodes ++= window.slice(0, i)
          linkNodes ++= window.slice(i + 1, window.length)
          wordsLinks.put(word, (wordsLinks(word) | linkNodes) - word)
        }
      })
    })

    wordsLinks.map(i => i._1 -> i._2.toArray).toMap
  }

  val weightDenominator = conf.weighConf.filter(_._2 > 0).map(_._2).sum.toFloat
  val weiConf = conf.weighConf.filter(_._2 > 0).mapValues(_ / weightDenominator)

  def calcWeight(w1: String, w2: String): Double = {
    weiConf.map(e => e._2 * wFunMap(e._1)(w1, w2)).sum
  }

  var iterCnt = 0

  def calcScore(): Array[(String, Double)] = {
    var score = Map[String, Double]().withDefaultValue(1.0)
    var maxScoreDif = Double.MaxValue
    var i = 0
    while (maxScoreDif > conf.minScoreDif && i < conf.iter && !wordGraph.isEmpty) {
      var newScore = wordGraph.map(wlink => {
        val w1 = wlink._1
        val w1score = wlink._2.map(w2 => {
          score(w2) * calcWeight(w2, w1)
        }).sum
        (w1, (1 - conf.damp) / wordCnt + conf.damp * w1score)
      })
      maxScoreDif = newScore.map(e => math.abs(e._2 - score(e._1))).max
      score = newScore
      i += 1
    }
    iterCnt = i
    score.toArray.sortWith(_._2 > _._2).take(conf.topN)
  }



}
