const Esprima = require("esprima");
const Styx = require("styx");
const code = process.argv[2];

try {
    const ast = Esprima.parseScript(code);
    const flowProgram = Styx.parse(ast);
    const json = Styx.exportAsJson(flowProgram);
    console.log(JSON.stringify(json));
} catch (error) {
    // Output structured error message
    console.error(JSON.stringify({ error: true, message: error.message }));
}
