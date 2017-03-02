import java.awt.{Color, Image}
import java.awt.image.BufferedImage
import com.microsoft.CNTK.{Function => CNTKFunction, _}


object Utils {
  def seqify(fvv: FloatVectorVector): Seq[Seq[Float]] = {
    (0 until fvv.size.toInt).map(i =>
        (0 until fvv.get(i.toInt).size().toInt).map(j => fvv.get(i.toInt).get(j.toInt)))
  }

  def preprocessImage(img: BufferedImage) = {
    for {
      c <- 0 until 3
      h <- 0 until img.getHeight()
      w <- 0 until img.getWidth()
      color = new Color(img.getRGB(w, h))
      intensity = if (c == 0) {
        color.getBlue
      } else if (c == 1) {
        color.getGreen
      } else {
        color.getRed
      }
    } yield intensity.toFloat
  }
}
