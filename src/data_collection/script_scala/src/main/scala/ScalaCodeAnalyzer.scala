import scala.io.Source
import scala.util.Using

object ScalaCodeAnalyzer {

  def main(args: Array[String]): Unit = {
    if (args.length < 2) {
      println("Usage: ScalaCodeAnalyzer <filename> <line number>")
      return
    }

    val lineNumber = args(0).toInt
    val filename = args(1)


    val lines = Using(Source.fromFile(filename))(_.getLines().toList).getOrElse {
      println(s"Could not open file: $filename")
      return
    }

    if (lineNumber < 1 || lineNumber > lines.size) {
      println(s"$lineNumber $lineNumber")
      return
    }

    val lineContent = lines(lineNumber - 1)
    val (start, end) = findFunctionOrClassBounds(lines, lineNumber - 1)

    println(s"${start + 1} ${end + 1}")
  }

  def findFunctionOrClassBounds(lines: List[String], lineIndex: Int): (Int, Int) = {
    var startIndex = lineIndex
    var endIndex = lineIndex

    while (startIndex > 0 && !isFunctionOrClassStart(lines(startIndex))) {
      startIndex -= 1
    }

    while (endIndex < lines.size - 1 && !isFunctionOrClassEnd(lines(endIndex + 1))) {
      endIndex += 1
    }

    (startIndex, endIndex)
  }

  def isFunctionOrClassStart(line: String): Boolean = {
    line.trim.matches("(?i)(class|trait|object|def)\\s+.*")
  }

  def isFunctionOrClassEnd(line: String): Boolean = {
    line.trim.matches("^(}?\\s*\\n?)(//.*)?$") || line.trim.isEmpty
  }
}