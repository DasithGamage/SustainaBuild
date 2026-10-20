// Compare the browser engine (frontend/engine.js) with the Python simulator.
//   python export_web_engine.py <build> --reference
//   node review/check_engine.mjs <build>
// Every controller, every day, every scenario in reference.json: hourly actions must match exactly and
// the daily objective, bill, hourly temperatures and battery charge must agree to 1e-9.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import {fileURLToPath} from 'node:url';

const build = process.argv[2];
const here = path.dirname(fileURLToPath(import.meta.url));
vm.runInThisContext(fs.readFileSync(path.join(here, '..', 'frontend', 'engine.js'), 'utf8'));
const data = JSON.parse(fs.readFileSync(path.join(build, 'assets', 'engine-data.json'), 'utf8'));
const reference = JSON.parse(fs.readFileSync(path.join(build, 'reference.json'), 'utf8'));
const E = globalThis.SB_ENGINE._internal;
E.setData(data);

let runs = 0, actionMismatch = 0, valueMismatch = 0, worst = 0;
const examples = [];
for (const ref of reference) {
  const s = E.scenarioOf(ref.scenario), cfg = E.configOf(s), day = data.days.find(d => d.date === s.date);
  for (const [name, py] of Object.entries(ref.results)) {
    const js = E.runDay(name, day, cfg);
    runs++;
    const acts = js.trace.map(r => r.action);
    if (acts.some((a, i) => a !== py.actions[i])) {
      actionMismatch++;
      if (examples.length < 8) examples.push({name, scenario: s, js: acts.join(','), py: py.actions.join(',')});
      continue;
    }
    const diffs = [Math.abs(js.metrics.objective - py.objective), Math.abs(js.metrics.energy_cost - py.energy_cost),
      ...js.trace.map((r, i) => Math.abs(r.Ta - py.Ta[i])), ...js.trace.map((r, i) => Math.abs(r.soc - py.soc[i]))];
    const d = Math.max(...diffs);
    worst = Math.max(worst, d);
    if (d > 1e-9) valueMismatch++;
  }
}
console.log(`${runs} controller-days compared (${reference.length} day-scenarios x 5 controllers)`);
console.log(`hourly action sequences that differ: ${actionMismatch}`);
console.log(`runs with metric/trace difference > 1e-9: ${valueMismatch} (largest difference ${worst.toExponential(2)})`);
for (const e of examples) console.log('MISMATCH', JSON.stringify(e));
process.exit(actionMismatch || valueMismatch ? 1 : 0);
