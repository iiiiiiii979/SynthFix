const Esprima = require("esprima");
const Styx = require("styx");


var code = "var x = 1;";
var ast = Esprima.parse(code);
var flowProgram = Styx.parse(ast);
var json = Styx.exportAsJson(flowProgram);

console.log(json);
