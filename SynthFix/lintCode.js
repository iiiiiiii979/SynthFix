// lintCode.js
const { ESLint } = require("eslint");

async function lintCode(text) {
    const eslint = new ESLint();
    const results = await eslint.lintText(text);
    console.log(JSON.stringify(results));
}

const code = process.argv[2];
lintCode(code);