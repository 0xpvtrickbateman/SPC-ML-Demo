// Test the shipped helper's form validation and generated import, without a browser download.
import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import vm from 'node:vm';
const html=readFileSync(new URL('../dashboards/setup.html',import.meta.url),'utf8');
const script=html.split('<script>')[1].split('</script>')[0];
const elements={};
for(const id of ['catalog','schema','download','validation','config'])elements[id]={value:id==='schema'?'demo_schema':id==='catalog'?'ml_statistical_process_controls':'',disabled:id==='download',listeners:{},addEventListener(event,fn){this.listeners[event]=fn;}};
let saved;
const context={document:{getElementById:id=>elements[id],body:{appendChild(){}},createElement:()=>({click(){},remove(){}})},Blob,URL:{createObjectURL(blob){saved=blob;return 'blob:test';},revokeObjectURL(){}},setTimeout(){}};
vm.runInNewContext(script,context);
assert.equal(elements.download.disabled,false);
assert.match(elements.config.textContent,/ml_statistical_process_controls.demo_schema/);
elements.catalog.value='attainx_demo';elements.catalog.listeners.input();
assert.equal(elements.download.disabled,false);
assert.match(elements.config.textContent,/OUTPUT_SCHEMA = "attainx_demo.demo_schema"/);
elements.download.listeners.click();
const dashboard=JSON.parse(await saved.text());
assert.equal(dashboard.datasets.length,10);
for(const d of dashboard.datasets){assert.ok(d.queryLines.join('').includes('attainx_demo.demo_schema.'));assert.ok(!d.queryLines.join('').includes('ml_statistical_process_controls.demo_schema.'));}
elements.catalog.value='invalid; sql';elements.catalog.listeners.input();assert.equal(elements.download.disabled,true);
const priorDownload=saved;elements.download.listeners.click();assert.equal(saved,priorDownload);
elements.catalog.value='valid_catalog';elements.schema.value='bad.schema';elements.schema.listeners.input();assert.equal(elements.download.disabled,true);
elements.schema.value='custom_schema';elements.schema.listeners.input();assert.equal(elements.download.disabled,false);
elements.download.listeners.click();
for(const d of JSON.parse(await saved.text()).datasets)assert.ok(d.queryLines.join('').includes('valid_catalog.custom_schema.'));
console.log('PASS: setup helper validates names, displays notebook setting and configures all ten dataset queries.');
