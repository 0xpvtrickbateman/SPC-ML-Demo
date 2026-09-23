// Test the shipped helper's form validation and generated import, without a browser download.
import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import vm from 'node:vm';
const html=readFileSync(new URL('../dashboards/setup.html',import.meta.url),'utf8');
const script=html.split('<script>')[1].split('</script>')[0];
const elements={};
for(const id of ['catalog','schema','download','validation','config'])elements[id]={value:id==='schema'?'spc_demo':'',disabled:id==='download',listeners:{},addEventListener(event,fn){this.listeners[event]=fn;}};
let saved;
const context={document:{getElementById:id=>elements[id],body:{appendChild(){}},createElement:()=>({click(){},remove(){}})},Blob,URL:{createObjectURL(blob){saved=blob;return 'blob:test';},revokeObjectURL(){}},setTimeout(){}};
vm.runInNewContext(script,context);
elements.catalog.value='attainx_demo';elements.catalog.listeners.input();
assert.equal(elements.download.disabled,false);
assert.match(elements.config.textContent,/OUTPUT_SCHEMA = "attainx_demo.spc_demo"/);
elements.download.listeners.click();
const dashboard=JSON.parse(await saved.text());
assert.equal(dashboard.datasets.length,10);
for(const d of dashboard.datasets){assert.ok(d.queryLines.join('').includes('attainx_demo.spc_demo.'));assert.ok(!d.queryLines.join('').includes('demo_catalog.demo_schema.'));}
elements.catalog.value='invalid; sql';elements.catalog.listeners.input();assert.equal(elements.download.disabled,true);
console.log('PASS: setup helper validates names, displays notebook setting and configures all ten dataset queries.');
