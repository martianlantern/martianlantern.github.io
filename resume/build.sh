#!/bin/bash
# Build CV and copy to assets/pdfs

cd "$(dirname "$0")"

echo "Compiling CV..."
pdflatex cv.tex > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "Copying to assets/pdfs/cv.pdf"
    cp cv.pdf ../assets/pdfs/cv.pdf
    echo "Done!"
else
    echo "Error: LaTeX compilation failed"
    pdflatex cv.tex
    exit 1
fi
