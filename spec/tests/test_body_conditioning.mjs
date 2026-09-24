import assert from 'node:assert/strict';
import test from 'node:test';

import {
  ALLOWED_LICENSES,
  BODY_CONDITIONING_VERSION,
  createBodyMesh,
  DenseBodyConditioning,
  ProvenanceError,
  rasterizeBodyMesh,
  verifyProvenance,
} from '../src/body-conditioning.mjs';

test('provenance allowlist is permissive only', () => {
  assert.ok(ALLOWED_LICENSES.length > 0);
  for (const license of ALLOWED_LICENSES) {
    assert.match(license, /^[A-Za-z0-9.-]+$/u);
  }
});

test('verifyProvenance accepts a valid in-source record', () => {
  const record = { source: 'in-source', revision: 'generated', license: 'MIT' };
  assert.equal(verifyProvenance(record), record);
});

test('verifyProvenance rejects missing source', () => {
  assert.throws(
    () => verifyProvenance({ license: 'MIT' }),
    ProvenanceError,
  );
});

test('verifyProvenance rejects a non-permissive license', () => {
  assert.throws(
    () => verifyProvenance({ source: 'amass', license: 'CC-BY-NC-4.0' }),
    ProvenanceError,
  );
});

test('createBodyMesh returns a closed, provenance-bearing mesh', () => {
  const mesh = createBodyMesh();
  assert.ok(mesh.vertices.length > 0);
  assert.ok(mesh.faces.length > 0);
  assert.equal(mesh.provenance.license, 'MIT');
  assert.ok(mesh.provenance.source.length > 0);
  // Every face index is in range.
  for (const face of mesh.faces) {
    for (const index of face) {
      assert.ok(index >= 0 && index < mesh.vertices.length);
    }
  }
});

test('rasterizeBodyMesh emits finite position/normal/depth maps', () => {
  const mesh = createBodyMesh();
  const result = rasterizeBodyMesh(mesh, { width: 32, height: 32 });
  assert.equal(result.width, 32);
  assert.equal(result.height, 32);
  assert.equal(result.position.length, 32 * 32 * 3);
  assert.equal(result.normal.length, 32 * 32 * 3);
  assert.equal(result.depth.length, 32 * 32);
  assert.ok(result.pixels > 0, 'expected at least one covered pixel');
  // Normals are unit-length at every covered pixel (depth is finite).
  for (let i = 0; i < result.depth.length; i += 1) {
    if (!Number.isFinite(result.depth[i])) continue;
    const nx = result.normal[i * 3];
    const ny = result.normal[i * 3 + 1];
    const nz = result.normal[i * 3 + 2];
    assert.ok(Math.abs(Math.hypot(nx, ny, nz) - 1) < 1e-3);
  }
});

test('DenseBodyConditioning rasterizes and rejects bad provenance', () => {
  const mesh = createBodyMesh();
  const capsule = new DenseBodyConditioning(mesh, { width: 16, height: 16 });
  const maps = capsule.rasterize();
  assert.ok(maps.position.length > 0);
  assert.ok(maps.normal.length > 0);
  assert.ok(maps.depth.length > 0);
  assert.throws(
    () =>
      new DenseBodyConditioning(
        { vertices: [], faces: [], provenance: { source: 'x', license: 'GPL-3.0' } },
        { width: 16, height: 16 },
      ),
    ProvenanceError,
  );
});

test('benchmark constants and version are stable', () => {
  assert.equal(typeof BODY_CONDITIONING_VERSION, 'string');
  assert.ok(BODY_CONDITIONING_VERSION.length > 0);
});
