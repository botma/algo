package linj.common

import com.github.fommil.netlib.F2jBLAS

/**
  * Created by linjiang on 2017/4/19.
  */
object Blas {
  private val f2jBlas = new F2jBLAS

  def dot(x:Array[Double],y:Array[Double]):Double = {
    f2jBlas.ddot(x.length,x,1,y,1)
  }
  def cos(x:Array[Double],y:Array[Double]):Double = {
    dot(x,y)/(math.sqrt(dot(x,x)) * math.sqrt(dot(y,y)))
  }
  def dot(x:Array[Float],y:Array[Float]):Double = {
    f2jBlas.sdot(x.length,x,1,y,1)
  }
  def norm2(x:Array[Float])={
    val n = f2jBlas.snrm2(x.length,x,1)
    for(i <- 0 until x.length){
      x(i) = x(i) / n
    }
    x
  }
  def cos(x:Array[Float],y:Array[Float]):Double = {
    dot(x,y)/(math.sqrt(dot(x,x)) * math.sqrt(dot(y,y)))
  }

  def main(args: Array[String]): Unit = {
    val x =Array(1.0f,1.0f)
    val f = norm2(x)
    println(x.mkString(","))
    val dot = f2jBlas.sdot(f.length,f,1,f,1)
    println(dot)

    val a=Array(1.0f,1.0f)
    val b=Array(2.0f,3.0f)
    f2jBlas.saxpy(a.length,-1f,a,1,b,1)
    println(b.mkString(","))
  }
}
