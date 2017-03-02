name := "ScalaEvalExample"

version := "1.0"

scalaVersion := "2.11.8"

val sparkVer = "2.0.0"

fork := true

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core"  % sparkVer,
  "org.apache.spark" %% "spark-mllib" % sparkVer
)
unmanagedJars in Compile += file("lib/cntk.jar")

javaOptions += "-Djava.library.path=C:/repos/cntk/x64/Release"

