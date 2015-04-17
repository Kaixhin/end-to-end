# End-to-End Training of Deep Visuomotor Policies

Presentation on paper.

## Build

Run `git submodule update --init` to fetch reveal.js.
Change into its directory and use `npm install` to fetch its dependencies.
Run `grunt --force` to build the reveal.js minimised files.
Run `pandoc -t revealjs --standalone --self-contained -s pres.md -o pres.html` to build the presentation.
