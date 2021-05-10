#!/bin/bash

version=`git describe --abbrev=40 --dirty --always --tags 2>/dev/null`
if [[ "${version}" == "" ]]; then
	version=`cat BASE_VERSION`"-release"
fi
echo $version
