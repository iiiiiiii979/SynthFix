const esprima = require('esprima');
const esgraph = require('esgraph');
const fs = require('fs');

const code = `
function clearAllData(wgt) {
    wgt._clearSelection();
    wgt._separatorCode = wgt._startOnSearching = wgt._chgSel = wgt.fixInputWidth = null;
}
function startOnSearching(wgt) {
    if (!wgt._startOnSearching)
        wgt._startOnSearching = setTimeout(function () {
            wgt._fireOnSearching(wgt.$n(('inp')).value);
            wgt._startOnSearching = null;
        }, 350);
}`;

const ast = esprima.parseScript(code);

const cfgs = ast.body.map(func => esgraph(func.body)); 

cfgs.forEach((cfg, index) => {
    console.log(`CFG for Function ${index}:`);
    console.log(cfg);
    
    cfg[2].forEach(edge => {
        console.log(`Node ${edge.from.uid} -> Node ${edge.to.uid}`);
    });
});
