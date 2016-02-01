#!/bin/bash

tempfile=$(mktemp /tmp/dirtree-XXXXX.tex)
trap "rm $tempfile" 0 1 2 3 6 9 15

cat <<EOF >$tempfile
\documentclass[11pt,a4paper,oneside]{article}
\usepackage{fullpage,verbatim,dirtree}
\begin{document}
\section{Listing}
\dirtree{%
EOF

export -a scriptList=()
while IFS=/ read -a fPath ;do
    file="${fPath[*]:${#fPath[*]}-1}"
    IFS=/
    full="${fPath[*]}"
    type="$(file -b "$full")"
#    echo .${#fPath[@]} "${file//_/\\_}\DTcomment{$type}. "
    echo .${#fPath[@]} "${file//_/\\_}. "
    [[ "$type" =~ script.text ]] && scriptList=("${scriptList[@]}" "$full")
    done  < <(
    find $1 -type d -o -type f
)  >>$tempfile

export IFS=$'\n\t '
echo "}" >>$tempfile

for file in "${scriptList[@]}";do
    name="${file##*/}"
    printf "\\section{%s}\n{\\\\scriptsize\\\\verbatiminput{%s}}\n" \
    "${name//_/\_}" "${file}"  >>"${tempfile}"    
done

echo >>"${tempfile}" '\end{document}'

#pdflatex -interaction nonstopmode "${tempfile}"

cat ${tempfile}
