#!/usr/bin/env bash
# Publish the terrarium (web/ with the data from export_web.py) to GitHub Pages:
# https://s31fdev.github.io/fly-terrarium/
# The site goes to the gh-pages branch as a single commit that replaces the previous one,
# so the 80 MB of data do not pile up in the history. Needs `gh` logged in.
set -euo pipefail
cd "$(dirname "$0")"
name=$(git config user.name)
email=$(git config user.email)
site=$(mktemp -d)
cp web/index.html web/app.js web/brain.js web/icon.svg "$site/"
cp web/SITE_README.md "$site/README.md"
cp LICENSE "$site/"
for v in 783 mcns; do
  mkdir -p "$site/data/$v"
  cp web/data/$v/connectome.bin.gz.* web/data/$v/map.bin web/data/$v/meta.json "$site/data/$v/"
done
touch "$site/.nojekyll" # plain files, no Jekyll build
cd "$site"
git init -q -b gh-pages
git add -A
git -c user.name="$name" -c user.email="$email" commit -q -m "Fly terrarium: static site"
git -c credential.helper= -c "credential.helper=!gh auth git-credential" \
  push -f https://github.com/s31fdev/fly-terrarium.git gh-pages
cd - > /dev/null
rm -rf "$site"
