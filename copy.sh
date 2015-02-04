#! /bin/sh

name=SwingingBrutusL
rm -rf $name
mkdir -p $name/src/Brutus
cp x64/Release/Brutus.dll $name/$name.dll
cp *.sln $name/src
cp Brutus/*.cpp Brutus/*.vcxproj Brutus/*.vcxproj.filters $name/src/Brutus


