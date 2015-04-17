Get reveal.js repo
Change into it and use npm install
Run grunt --force
Run pandoc -t revealjs --standalone --self-contained \
  -V theme=moon \
  -s habits.md -o habits.html
