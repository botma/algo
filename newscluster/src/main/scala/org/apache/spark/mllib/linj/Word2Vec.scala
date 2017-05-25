/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.linj

import java.lang.{Iterable => JavaIterable}

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.apache.spark.annotation.Since
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.rdd._
import org.apache.spark.sql.SQLContext
import org.apache.spark.util.Utils
import org.apache.spark.util.random.XORShiftRandom
import org.apache.spark.{Logging, SparkContext}
import org.json4s.DefaultFormats
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuilder
import scala.reflect.ClassTag

/**
  * Entry in vocabulary
  */
private case class VocabWord(
                              var word: String,
                              var cn: Int,
                              var point: Array[Int],
                              var code: Array[Int],
                              var codeLen: Int
                            )

private class WordVecConf extends  Serializable{
  var vectorSize = 100
  var learningRate = 0.025
  var MIN_LR = learningRate * 0.0001
  var numPartitions = 1
  var numIterations = 1
  var seed = Utils.random.nextLong()
  var minCount = 5

  val EXP_TABLE_SIZE = 1000
  val MAX_EXP = 6
  val MAX_CODE_LENGTH = 40
  val MAX_SENTENCE_LENGTH = 1000

  /** context words from [-window, window] */
  var window = 5

}

/**
  * Word2Vec creates vector representation of words in a text corpus.
  * The algorithm first constructs a vocabulary from the corpus
  * and then learns vector representation of words in the vocabulary.
  * The vector representation can be used as features in
  * natural language processing and machine learning algorithms.
  *
  * We used skip-gram model in our implementation and hierarchical softmax
  * method to train the model. The variable names in the implementation
  * matches the original C implementation.
  *
  * For original C implementation, see https://code.google.com/p/word2vec/
  * For research papers, see
  * Efficient Estimation of Word Representations in Vector Space
  * and
  * Distributed Representations of Words and Phrases and their Compositionality.
  */
@Since("1.1.0")
class Word2Vec extends Logging/*extends Serializable with Logging*/ {

  private val conf = new WordVecConf

  /**
    * Sets vector size (default: 100).
    */
  @Since("1.1.0")
  def setVectorSize(vectorSize: Int): this.type = {
    this.conf.vectorSize = vectorSize
    this
  }

  /**
    * Sets initial learning rate (default: 0.025).
    */
  @Since("1.1.0")
  def setLearningRate(learningRate: Double): this.type = {
    this.conf.learningRate = learningRate
    this.conf.MIN_LR = learningRate * 0.0001
    this
  }

  /**
    * Sets number of partitions (default: 1). Use a small number for accuracy.
    */
  @Since("1.1.0")
  def setNumPartitions(numPartitions: Int): this.type = {
    require(numPartitions > 0, s"numPartitions must be greater than 0 but got $numPartitions")
    this.conf.numPartitions = numPartitions
    this
  }

  /**
    * Sets number of iterations (default: 1), which should be smaller than or equal to number of
    * partitions.
    */
  @Since("1.1.0")
  def setNumIterations(numIterations: Int): this.type = {
    this.conf.numIterations = numIterations
    this
  }

  /**
    * Sets random seed (default: a random long integer).
    */
  @Since("1.1.0")
  def setSeed(seed: Long): this.type = {
    this.conf.seed = seed
    this
  }

  /**
    * Sets the window of words (default: 5)
    */
  @Since("1.6.0")
  def setWindowSize(window: Int): this.type = {
    this.conf.window = window
    this
  }

  /**
    * Sets minCount, the minimum number of times a token must appear to be included in the word2vec
    * model's vocabulary (default: 5).
    */
  @Since("1.3.0")
  def setMinCount(minCount: Int): this.type = {
    this.conf.minCount = minCount
    this
  }


  private var trainWordsCount = 0L
  private var vocabSize = 0
  @transient private var vocab: Array[VocabWord] = null
  @transient private var vocabHash = mutable.HashMap.empty[String, Int]

  private def learnVocab(words: RDD[String]): Unit = {
    val conf = this.conf
    vocab = words.map(w => (w, 1))
      .reduceByKey(_ + _)
      .filter(_._2 >= conf.minCount)
      .map(x => VocabWord(
        x._1,
        x._2,
        new Array[Int](conf.MAX_CODE_LENGTH),
        new Array[Int](conf.MAX_CODE_LENGTH),
        0))
      .collect()
      .sortWith((a, b) => a.cn > b.cn)

    vocabSize = vocab.length
    require(vocabSize > 0, "The vocabulary size should be > 0. You may need to check " +
      "the setting of minCount, which could be large enough to remove all your words in sentences.")

    var a = 0
    while (a < vocabSize) {
      vocabHash += vocab(a).word -> a
      trainWordsCount += vocab(a).cn
      a += 1
    }
    logInfo(s"vocabSize = $vocabSize, trainWordsCount = $trainWordsCount")
  }

  private def createExpTable(): Array[Float] = {
    val expTable = new Array[Float](conf.EXP_TABLE_SIZE)
    var i = 0
    while (i < conf.EXP_TABLE_SIZE) {
      val tmp = math.exp((2.0 * i / conf.EXP_TABLE_SIZE - 1.0) * conf.MAX_EXP)
      expTable(i) = (tmp / (tmp + 1.0)).toFloat
      i += 1
    }
    expTable
  }

  private def createBinaryTree(): Unit = {
    val count = new Array[Long](vocabSize * 2 + 1)
    val binary = new Array[Int](vocabSize * 2 + 1)
    val parentNode = new Array[Int](vocabSize * 2 + 1)
    val code = new Array[Int](conf.MAX_CODE_LENGTH)
    val point = new Array[Int](conf.MAX_CODE_LENGTH)
    var a = 0
    while (a < vocabSize) {
      count(a) = vocab(a).cn
      a += 1
    }
    while (a < 2 * vocabSize) {
      count(a) = 1e9.toInt
      a += 1
    }
    var pos1 = vocabSize - 1
    var pos2 = vocabSize

    var min1i = 0
    var min2i = 0

    a = 0
    while (a < vocabSize - 1) {
      if (pos1 >= 0) {
        if (count(pos1) < count(pos2)) {
          min1i = pos1
          pos1 -= 1
        } else {
          min1i = pos2
          pos2 += 1
        }
      } else {
        min1i = pos2
        pos2 += 1
      }
      if (pos1 >= 0) {
        if (count(pos1) < count(pos2)) {
          min2i = pos1
          pos1 -= 1
        } else {
          min2i = pos2
          pos2 += 1
        }
      } else {
        min2i = pos2
        pos2 += 1
      }
      count(vocabSize + a) = count(min1i) + count(min2i)
      parentNode(min1i) = vocabSize + a
      parentNode(min2i) = vocabSize + a
      binary(min2i) = 1
      a += 1
    }
    // Now assign binary code to each vocabulary word
    var i = 0
    a = 0
    while (a < vocabSize) {
      var b = a
      i = 0
      while (b != vocabSize * 2 - 2) {
        code(i) = binary(b)
        point(i) = b
        i += 1
        b = parentNode(b)
      }
      vocab(a).codeLen = i
      vocab(a).point(0) = vocabSize - 2
      b = 0
      while (b < i) {
        vocab(a).code(i - b - 1) = code(b)
        vocab(a).point(i - b) = point(b) - vocabSize
        b += 1
      }
      a += 1
    }
  }

  /**
    * Computes the vector representation of each word in vocabulary.
    *
    * @param dataset an RDD of words
    * @return a Word2VecModel
    */
  @Since("1.1.0")
  def fit[S <: Iterable[String]](dataset: RDD[S]): Array[(String, Array[Float])] = {

    val conf = this.conf

    logInfo(s"partitions=${conf.numPartitions}, window=${conf.window}, iter=${conf.numIterations}, mincount=${conf.minCount} ")

    val words = dataset.flatMap(x => x)

    logInfo("start learn Vocab")
    learnVocab(words)

    logInfo("start learn binary tree")
    createBinaryTree()

    val sc = dataset.context

    logInfo("start vec learning .......................")
    val expTable = sc.broadcast(createExpTable())
    val bcVocab = sc.broadcast(vocab)
    val bcVocabHash = sc.broadcast(vocabHash)

    val sentences: RDD[Array[Int]] = words.mapPartitions { iter =>
      new Iterator[Array[Int]] {
        def hasNext: Boolean = iter.hasNext

        def next(): Array[Int] = {
          val sentence = ArrayBuilder.make[Int]
          var sentenceLength = 0
          while (iter.hasNext && sentenceLength < conf.MAX_SENTENCE_LENGTH) {
            val word = bcVocabHash.value.get(iter.next())
            word match {
              case Some(w) =>
                sentence += w
                sentenceLength += 1
              case None =>
            }
          }
          sentence.result()
        }
      }
    }

    val newSentences = sentences.repartition(conf.numPartitions).cache()
    val initRandom = new XORShiftRandom(conf.seed)

    val vocabSize = this.vocabSize
    val trainWordsCount =this.trainWordsCount

    if (vocabSize.toLong * conf.vectorSize * 8 >= Int.MaxValue) {
      throw new RuntimeException("Please increase minCount or decrease vectorSize in Word2Vec" +
        " to avoid an OOM. You are highly recommended to make your vocabSize*vectorSize, " +
        "which is " + vocabSize + "*" + conf.vectorSize + " for now, less than `Int.MaxValue/8`.")
    }

    val syn0Global = 
      HighMatrix.fill[Float](vocabSize, conf.vectorSize)((initRandom.nextFloat() - 0.5f) / conf.vectorSize)
    val syn1Global =
      HighMatrix.fill[Float](vocabSize, conf.vectorSize)((initRandom.nextFloat() - 0.5f) / conf.vectorSize)


    for (k <- 1 to conf.numIterations) {

      var alpha = conf.learningRate / k

      logInfo(s"start iter $k ..............")

      val syn0GlobalBroad = sc.broadcast(syn0Global)
      val syn1GlobalBroad = sc.broadcast(syn1Global)

      val partial = newSentences.mapPartitionsWithIndex { case (idx, iter) =>
        val sync0Local = syn0GlobalBroad.value
        val sync1Local = syn1GlobalBroad.value

        val random = new XORShiftRandom(conf.seed ^ ((idx + 1) << 16) ^ ((-k - 1) << 8))
        val syn0Modify = new mutable.BitSet(vocabSize)
        val syn1Modify = new mutable.BitSet(vocabSize)
        var wordCount = 0L
        var lastWordCount = 0L

        iter.foreach(sentence => {

          if (alpha > conf.MIN_LR && wordCount - lastWordCount > 10000) {
            lastWordCount = wordCount
            alpha = conf.learningRate * (1 - wordCount.toDouble / (trainWordsCount + 1))
            if (alpha < conf.MIN_LR) alpha = conf.MIN_LR
          }

          wordCount += sentence.size
          var pos = 0
          while (pos < sentence.size) {
            val word = sentence(pos)
            val b = random.nextInt(conf.window)
            // Train Skip-gram
            var a = b
            while (a < conf.window * 2 + 1 - 2 * b) {
              if (a != conf.window) {
                val c = pos - conf.window + a
                if (c >= 0 && c < sentence.size) {
                  val lastWord = sentence(c)
                  val syn0 = sync0Local(lastWord)

                  val neu1e = new Array[Float](conf.vectorSize)
                  // Hierarchical softmax
                  var d = 0
                  while (d < bcVocab.value(word).codeLen) {
                    val inner = bcVocab.value(word).point(d)
                    // Propagate hidden -> output
                    val syn1 = sync1Local(inner)
                    var f = blas.sdot(conf.vectorSize, syn0._1, syn0._2, 1, syn1._1, syn1._2, 1)
                    if (- conf.MAX_EXP < f && f < conf.MAX_EXP) {
                      val ind = ((f + conf.MAX_EXP) * (conf.EXP_TABLE_SIZE / conf.MAX_EXP / 2.0)).toInt
                      f = expTable.value(ind)
                      val g = ((1 - bcVocab.value(word).code(d) - f) * alpha).toFloat
                      blas.saxpy(conf.vectorSize, g, syn1._1, syn1._2, 1, neu1e, 0, 1)
                      blas.saxpy(conf.vectorSize, g, syn0._1, syn0._2, 1, syn1._1, syn1._2, 1)
                      syn1Modify += inner
                    }
                    d += 1
                  }
                  blas.saxpy(conf.vectorSize, 1.0f, neu1e, 0, 1, syn0._1, syn0._2, 1)
                  syn0Modify += lastWord
                }
              }
              a += 1
            }
            pos += 1
          }
        })

        syn0Modify.map(ind => (ind, (sync0Local.get(ind), 1))).toIterator ++
          syn1Modify.map(ind => (ind + vocabSize, (sync1Local.get(ind), 1))).toIterator
      }

      logInfo("start merge vec............")
      var syn0Cnt = 0L
      var syn1Cnt = 1L
      val updateRdd = partial.reduceByKey { case (v1, v2) =>
        blas.saxpy(conf.vectorSize, 1.0f, v2._1, 1, v1._1, 1)
        (v1._1, v1._2 + v2._2)
      }
      logInfo("start collect vec............")

      try {
        val collect = updateRdd.collect()//.toLocalIterator

        logInfo("start update global vec............")
        collect.foreach(item => {
          val index = item._1
          //        val v = item._2
          val vec = item._2._1
          val cnt = item._2._2
          if (index < vocabSize) {
            syn0Global.update(index, vec)
            syn0Cnt += 1
          } else {
            syn1Global.update(index - vocabSize, vec)
            syn1Cnt += 1
          }
        })
      } finally {
        updateRdd.unpersist(false)
      }
      syn0GlobalBroad.unpersist(true)
      syn1GlobalBroad.unpersist(true)
      logInfo(s"syn0 update = $syn0Cnt,syn1 update = $syn1Cnt  ......")

    }
    logInfo("finish fit ..............")

    newSentences.unpersist()
    expTable.destroy()
    bcVocab.destroy()
    bcVocabHash.destroy()
    vocab.zipWithIndex.map(zi => {
      val ind = zi._2
      val word = zi._1.word
      (word, syn0Global.get(ind))
    })
  }

}

/*
/**
  * Word2Vec model
  * @param wordIndex maps each word to an index, which can retrieve the corresponding
  *                  vector from wordVectors
  * @param wordVectors array of length numWords * vectorSize, vector corresponding
  *                    to the word mapped with index i can be retrieved by the slice
  *                    (i * vectorSize, i * vectorSize + vectorSize)
  */
@Since("1.1.0")
class Word2VecModel private[spark] (
                                     private[spark] val wordIndex: Map[String, Int],
                                     private[spark] val wordVectors: HighMatrix[Float]) extends Serializable with Saveable {

  private val numWords = wordIndex.size
  // vectorSize: Dimension of each word's vector.
  private val vectorSize = wordVectors.length / numWords

  // wordList: Ordered list of words obtained from wordIndex.
  private val wordList: Array[String] = {
    val (wl, _) = wordIndex.toSeq.sortBy(_._2).unzip
    wl.toArray
  }

  // wordVecNorms: Array of length numWords, each value being the Euclidean norm
  //               of the wordVector.
  private val wordVecNorms: Array[Double] = {
    val wordVecNorms = new Array[Double](numWords)
    var i = 0
    while (i < numWords) {
      val vec = wordVectors(i)
      wordVecNorms(i) = blas.snrm2(vectorSize, vec, 1)
      i += 1
    }
    wordVecNorms
  }

  @Since("1.5.0")
  def this(model: Map[String, Array[Float]]) = {
    this(Word2VecModel.buildWordIndex(model), Word2VecModel.buildWordVectors(model))
  }

  private def cosineSimilarity(v1: Array[Float], v2: Array[Float]): Double = {
    require(v1.length == v2.length, "Vectors should have the same length")
    val n = v1.length
    val norm1 = blas.snrm2(n, v1, 1)
    val norm2 = blas.snrm2(n, v2, 1)
    if (norm1 == 0 || norm2 == 0) return 0.0
    blas.sdot(n, v1, 1, v2, 1) / norm1 / norm2
  }

  override protected def formatVersion = "1.0"

  @Since("1.4.0")
  def save(sc: SparkContext, path: String): Unit = {
    Word2VecModel.SaveLoadV1_0.save(sc, path, getVectors)
  }

  /**
    * Transforms a word to its vector representation
    * @param word a word
    * @return vector representation of word
    */
  @Since("1.1.0")
  def transform(word: String): Vector = {
    wordIndex.get(word) match {
      case Some(ind) =>
        val vec = wordVectors(ind)
        Vectors.dense(vec.map(_.toDouble))
      case None =>
        throw new IllegalStateException(s"$word not in vocabulary")
    }
  }

  /**
    * Find synonyms of a word
    * @param word a word
    * @param num number of synonyms to find
    * @return array of (word, cosineSimilarity)
    */
  @Since("1.1.0")
  def findSynonyms(word: String, num: Int): Array[(String, Double)] = {
    val vector = transform(word)
    findSynonyms(vector, num)
  }

  /**
    * Find synonyms of the vector representation of a word
    * @param vector vector representation of a word
    * @param num number of synonyms to find
    * @return array of (word, cosineSimilarity)
    */
  @Since("1.1.0")
  def findSynonyms(vector: Vector, num: Int): Array[(String, Double)] = {
    require(num > 0, "Number of similar words should > 0")
    // TODO: optimize top-k
    val fVector = vector.toArray.map(_.toFloat)
    val cosineVec = Array.fill[Float](numWords)(0)
    val alpha: Float = 1
    val beta: Float = 0

    for(ind <- 0 until wordVectors.length){
      cosineVec(ind)=blas.sdot(vectorSize, wordVectors(ind),0,1, fVector,0, 1)
    }

    // Need not divide with the norm of the given vector since it is constant.
    val cosVec = cosineVec.map(_.toDouble)
    var ind = 0
    while (ind < numWords) {
      val norm = wordVecNorms(ind)
      if (norm == 0.0) {
        cosVec(ind) = 0.0
      } else {
        cosVec(ind) /= norm
      }
      ind += 1
    }
    wordList.zip(cosVec)
      .toSeq
      .sortBy(- _._2)
      .take(num + 1)
      .tail
      .toArray
  }

  /**
    * Returns a map of words to their vector representations.
    */
  @Since("1.2.0")
  def getVectors: Map[String, Array[Float]] = {
    wordIndex.map { case (word, ind) =>
      (word, wordVectors(ind))
    }
  }
}

@Since("1.4.0")
object Word2VecModel extends Loader[Word2VecModel] {

  private def buildWordIndex(model: Map[String, Array[Float]]): Map[String, Int] = {
    model.keys.zipWithIndex.toMap
  }

  private def buildWordVectors(model: Map[String, Array[Float]]): Array[Float] = {
    require(model.nonEmpty, "Word2VecMap should be non-empty")
    val (vectorSize, numWords) = (model.head._2.size, model.size)
    val wordList = model.keys.toArray
    val wordVectors = new Array[Float](vectorSize * numWords)
    var i = 0
    while (i < numWords) {
      Array.copy(model(wordList(i)), 0, wordVectors, i * vectorSize, vectorSize)
      i += 1
    }
    wordVectors
  }

  private object SaveLoadV1_0 {

    val formatVersionV1_0 = "1.0"

    val classNameV1_0 = "org.apache.spark.mllib.feature.Word2VecModel"

    case class Data(word: String, vector: Array[Float])

    def load(sc: SparkContext, path: String): Word2VecModel = {
      val dataPath = Loader.dataPath(path)
      val sqlContext = SQLContext.getOrCreate(sc)
      val dataFrame = sqlContext.read.parquet(dataPath)
      // Check schema explicitly since erasure makes it hard to use match-case for checking.
      Loader.checkSchema[Data](dataFrame.schema)

      val dataArray = dataFrame.select("word", "vector").collect()
      val word2VecMap = dataArray.map(i => (i.getString(0), i.getSeq[Float](1).toArray)).toMap
      new Word2VecModel(word2VecMap)
    }

    def save(sc: SparkContext, path: String, model: Map[String, Array[Float]]): Unit = {

      val sqlContext = SQLContext.getOrCreate(sc)
      import sqlContext.implicits._

      val vectorSize = model.values.head.size
      val numWords = model.size
      val metadata = compact(render
      (("class" -> classNameV1_0) ~ ("version" -> formatVersionV1_0) ~
        ("vectorSize" -> vectorSize) ~ ("numWords" -> numWords)))
      sc.parallelize(Seq(metadata), 1).saveAsTextFile(Loader.metadataPath(path))

      val dataArray = model.toSeq.map { case (w, v) => Data(w, v) }
      sc.parallelize(dataArray.toSeq, 1).toDF().write.parquet(Loader.dataPath(path))
    }
  }

  @Since("1.4.0")
  override def load(sc: SparkContext, path: String): Word2VecModel = {

    val (loadedClassName, loadedVersion, metadata) = Loader.loadMetadata(sc, path)
    implicit val formats = DefaultFormats
    val expectedVectorSize = (metadata \ "vectorSize").extract[Int]
    val expectedNumWords = (metadata \ "numWords").extract[Int]
    val classNameV1_0 = SaveLoadV1_0.classNameV1_0
    (loadedClassName, loadedVersion) match {
      case (classNameV1_0, "1.0") =>
        val model = SaveLoadV1_0.load(sc, path)
        val vectorSize = model.getVectors.values.head.size
        val numWords = model.getVectors.size
        require(expectedVectorSize == vectorSize,
          s"Word2VecModel requires each word to be mapped to a vector of size " +
            s"$expectedVectorSize, got vector of size $vectorSize")
        require(expectedNumWords == numWords,
          s"Word2VecModel requires $expectedNumWords words, but got $numWords")
        model
      case _ => throw new Exception(
        s"Word2VecModel.load did not recognize model with (className, format version):" +
          s"($loadedClassName, $loadedVersion).  Supported:\n" +
          s"  ($classNameV1_0, 1.0)")
    }
  }
}
*/
private class HighMatrix[T](row: Int, column: Int)(implicit m: ClassTag[T]) extends Serializable {

  private val rowPerBlock: Int = HighMatrix.BLOCK_SIZE / column
  private val blockCnt = math.ceil(row.asInstanceOf[Float] / rowPerBlock).asInstanceOf[Int]

  private val elements = new Array[Array[T]](blockCnt): Array[Array[T]]
  init()

  private def init(): Unit = {
    if (row > rowPerBlock) {
      for (i <- 0 until elements.length - 1) {
        elements(i) = Array.ofDim(rowPerBlock * column)
      }
      elements(elements.length - 1) = Array.ofDim((row - rowPerBlock * (elements.length - 1)) * column)
    } else {
      elements(0) = Array.ofDim(row * column)
    }
  }

  def apply(rowIndex: Int): (Array[T], Int, Int) = {
    if (rowIndex > row || rowIndex < 0) throw new ArrayIndexOutOfBoundsException(rowIndex)
    val blockStart = (rowIndex % rowPerBlock) * column
    (elements(rowIndex / rowPerBlock), blockStart, blockStart + column)
  }

  def get(rowIndex: Int): Array[T] = {
    val loc = apply(rowIndex)
    loc._1.slice(loc._2, loc._3)
  }


  def update(row: Int, value: Array[T]): Unit = {
    if (value.length != this.column) throw new IllegalArgumentException("array length not fit")
    val a = apply(row)
    Array.copy(value, 0, a._1, a._2, column)
  }

  def length = row

}

private object HighMatrix {

  val BLOCK_SIZE = (1 << 10) * 200

  def fill[T: ClassTag](row: Int, column: Int)(elem: => T): HighMatrix[T] = {
    val ret = new HighMatrix[T](row, column)
    ret.elements.foreach(
      a2 => for (i <- 0 until a2.length) a2(i) = elem
    )
    ret
  }
}
