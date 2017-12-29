#!/bin/bash

set -e # exit on failure (same as -o errexit)

lsb_release -a
sudo apt-get -qq update

if [[ "$CC" == gcc-5 ]]; then
	sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
	sudo apt-get -qq update
	sudo apt-get install -qq g++-5
elif [[ "$CC" == gcc-6 ]]; then
	sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
	sudo apt-get -qq update
	sudo apt-get install -qq g++-6
fi

sudo apt-get -qq install libboost-dev

WITHLANG=$SWIGLANG

case "$SWIGLANG" in
	"")     ;;
	"csharp")
		sudo apt-get -qq install mono-devel
		;;
	"d")
		wget http://downloads.dlang.org/releases/2014/dmd_2.066.0-0_amd64.deb
		sudo dpkg -i dmd_2.066.0-0_amd64.deb
		;;
	"go")
		;;
	"javascript")
		case "$ENGINE" in
			"node")
				sudo add-apt-repository -y ppa:chris-lea/node.js
				sudo apt-get -qq update
				sudo apt-get install -qq nodejs rlwrap
				sudo npm install -g node-gyp
				;;
			"jsc")
				sudo apt-get install -qq libwebkitgtk-dev
				;;
			"v8")
				sudo apt-get install -qq libv8-dev
				;;
		esac
		;;
	"guile")
		sudo apt-get -qq install guile-2.0-dev
		;;
	"lua")
		if [[ -z "$VER" ]]; then
			sudo apt-get -qq install lua5.2 liblua5.2-dev
		else
			sudo add-apt-repository -y ppa:ubuntu-cloud-archive/mitaka-staging
			sudo apt-get -qq update
			sudo apt-get -qq install lua${VER} liblua${VER}-dev
		fi
		;;
	"ocaml")
		# configure also looks for ocamldlgen, but this isn't packaged.  But it isn't used by default so this doesn't matter.
		sudo apt-get -qq install ocaml ocaml-findlib
		;;
	"octave")
		if [[ -z "$VER" ]]; then
			sudo apt-get -qq install liboctave-dev
		else
			sudo add-apt-repository -y ppa:kwwette/octaves
			sudo apt-get -qq update
			sudo apt-get -qq install liboctave${VER}-dev
		fi
		;;
	"php5")
		sudo apt-get -qq install php5-cli php5-dev
		;;
	"php")
		sudo add-apt-repository -y ppa:ondrej/php
		sudo apt-get -qq update
		sudo apt-get -qq install php$VER-cli php$VER-dev
		;;
	"python")
		pip install pep8
		if [[ "$PY3" ]]; then
			sudo apt-get install -qq python3-dev
		fi
		WITHLANG=$SWIGLANG$PY3
		if [[ "$VER" ]]; then
			sudo add-apt-repository -y ppa:fkrull/deadsnakes
			sudo apt-get -qq update
			sudo apt-get -qq install python${VER}-dev
			WITHLANG=$SWIGLANG$PY3=$SWIGLANG$VER
		fi
		;;
	"r")
		sudo apt-get -qq install r-base
		;;
	"ruby")
		if [[ "$VER" ]]; then
			rvm install $VER
		fi
		;;
	"scilab")
		sudo apt-get -qq install scilab
		;;
	"tcl")
		sudo apt-get -qq install tcl-dev
		;;
esac

set +e # turn off exit on failure (same as +o errexit)
