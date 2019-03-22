#!/usr/bin/env bash

cd "$(dirname "$0")"

DS_2016="clickbait17-train-170331.zip"
DS_2017="clickbait17-train-170630.zip"
URL_BASE="http://www.uni-weimar.de/medien/webis/corpora/corpus-webis-clickbait-17/"

printf "Going to download some large files, press any key to continue..."
read

# Download / unzip 2016 dataset
curl -LO $URL_BASE$DS_2016 && unzip $DS_2016 && rm $DS_2016

# Download / unzip 2016 dataset
curl -LO $URL_BASE$DS_2017 && unzip $DS_2017 && rm $DS_2017
