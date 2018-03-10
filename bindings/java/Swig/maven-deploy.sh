#!/bin/bash
mvn gpg:sign-and-deploy-file -DpomFile=pom.xml -Dfile=cntk.jar -Durl=http://oss.sonatype.org/service/local/staging/deploy/maven2/ -DrepositoryId=ossrh
mvn gpg:sign-and-deploy-file -DpomFile=pom.xml -Dfile=cntk-sources.jar -Durl=http://oss.sonatype.org/service/local/staging/deploy/maven2/ -DrepositoryId=ossrh -Dclassifier=sources
mvn gpg:sign-and-deploy-file -DpomFile=pom.xml -Dfile=cntk-javadoc.jar -Durl=http://oss.sonatype.org/service/local/staging/deploy/maven2/ -DrepositoryId=ossrh -Dclassifier=javadoc
