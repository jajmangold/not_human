#!/usr/bin/env node
// Bounded benchmark for dense body conditioning (ADR-0003, Phase 8).
//
// Fixed reference pose (the default in-source mesh), fixed resolution, fixed
// iteration count — so the measurement is deterministic and finite. Prints a
// JSON result to stdout.

import {
  BODY_CONDITIONING_VERSION,
  createBodyMesh,
  DenseBodyConditioning,
} from '../src/body-conditioning.mjs';

const WIDTH = 64;
const HEIGHT = 64;
const ITERATIONS = 5;

const mesh = createBodyMesh();
const capsule = new DenseBodyConditioning(mesh, { width: WIDTH, height: HEIGHT });

// Warm-up (excluded from timing).
capsule.rasterize();

const start = process.hrtime.bigint();
let last = null;
for (let i = 0; i < ITERATIONS; i += 1) {
  last = capsule.rasterize();
}
const elapsedNs = Number(process.hrtime.bigint() - start);

let covered = 0;
for (let i = 0; i < last.depth.length; i += 1) {
  if (Number.isFinite(last.depth[i])) covered += 1;
}

const result = {
  version: BODY_CONDITIONING_VERSION,
  provenance: mesh.provenance,
  width: WIDTH,
  height: HEIGHT,
  iterations: ITERATIONS,
  wallMsPerIteration: elapsedNs / 1e6 / ITERATIONS,
  coveredPixels: covered,
  totalPixels: WIDTH * HEIGHT,
  maps: ['position', 'normal', 'depth'],
};

console.log(JSON.stringify(result, null, 2));
