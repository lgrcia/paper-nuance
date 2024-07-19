#/bin/bash

# today's date
today=$(date +'%Y-%m-%d')

# create submission directory
mkdir -p arxiv_$today

# copy figures
sh ./copy_figures.sh

# copy latex to arxiv directory
cp -r latex/* arxiv_$today

# compress the arxiv directory as a zip file
zip -r arxiv_$today.zip arxiv_$today

# remove the arxiv directory
# rm -rf arxiv_$today

