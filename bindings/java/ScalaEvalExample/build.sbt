name := "ScalaEvalExample"

version := "1.0"

scalaVersion := "2.11.8"

val sparkVer = "2.0.0"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core"  % sparkVer,
  "org.apache.spark" %% "spark-mllib" % sparkVer
)
