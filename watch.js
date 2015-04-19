var fs = require("fs");
var spawn = require('child_process').spawn;

fs.watchFile("pres.md", function () {
  spawn("pandoc", ["-t", "revealjs", "--standalone", "--slide-level", "1", "-s", "pres.md", "-o", "pres.html"]);
  console.log("Rebuilding presentation");
});
