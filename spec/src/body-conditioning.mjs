// License-neutral dense body conditioning for CanonicalBodyV1 (ADR-0003).
//
// The default dense primitive is a small procedurally generated body mesh
// (two stacked cylinders: torso and head). Because the geometry is generated
// in-source, its provenance is this file and its license is the repository
// license — no external artifact, no SMPL-X/AMASS dependency.
//
// The capsule rasterizes the mesh into per-pixel position, normal, and depth
// maps using a bounded z-buffer fill (per-pixel edge test), so the benchmark
// stays deterministic and finite.

export const BODY_CONDITIONING_VERSION = 'dense-body-conditioning-v1';

// Permissive license allowlist. Anything not on this list is rejected at
// artifact load time, which is how provenance + license status are enforced.
export const ALLOWED_LICENSES = Object.freeze([
  'MIT',
  'Apache-2.0',
  'BSD-2-Clause',
  'BSD-3-Clause',
  'ISC',
  'CC0-1.0',
  'Unlicense',
]);

export class ProvenanceError extends Error {
  constructor(message) {
    super(message);
    this.name = 'ProvenanceError';
  }
}

/**
 * Validate an artifact provenance record.
 *
 * @param {object} record
 * @param {string} record.source - where the artifact came from (file, repo+rev).
 * @param {string} record.license - SPDX identifier.
 * @param {string} [record.revision] - commit/revision; optional for in-source.
 * @returns {object} the same record, for chaining.
 * @throws {ProvenanceError} when source/license are missing or the license is
 *   not on the permissive allowlist.
 */
export function verifyProvenance(record) {
  if (record === null || typeof record !== 'object') {
    throw new ProvenanceError('provenance record is required');
  }
  if (typeof record.source !== 'string' || record.source.length === 0) {
    throw new ProvenanceError('provenance.source is required');
  }
  if (typeof record.license !== 'string' || record.license.length === 0) {
    throw new ProvenanceError('provenance.license is required');
  }
  if (!ALLOWED_LICENSES.includes(record.license)) {
    throw new ProvenanceError(
      `license "${record.license}" is not on the permissive allowlist`,
    );
  }
  return record;
}
/**
 * Generate the default license-neutral body mesh.
 *
 * @param {object} [options]
 * @param {number} [options.radialSegments=8] - segments per cylinder ring.
 * @param {number} [options.height=1.0] - total body height in meters.
 * @returns {{ vertices: number[][], faces: number[][], provenance: object }}
 */
export function createBodyMesh(options = {}) {
  const radial = Number.isInteger(options.radialSegments) && options.radialSegments >= 3
    ? options.radialSegments
    : 8;
  const height = Number.isFinite(options.height) && options.height > 0 ? options.height : 1.0;
  const vertices = [];
  const faces = [];
  const parts = [
    { centerZ: -0.33 * height, radius: 0.18 * height, halfHeight: 0.33 * height },
    { centerZ: 0.28 * height, radius: 0.12 * height, halfHeight: 0.16 * height },
  ];
  for (const part of parts) {
    const base = vertices.length;
    const bottom = [];
    const top = [];
    for (let i = 0; i < radial; i += 1) {
      const angle = (2 * Math.PI * i) / radial;
      const x = Math.cos(angle) * part.radius;
      const y = Math.sin(angle) * part.radius;
      bottom.push([x, y, part.centerZ - part.halfHeight]);
      top.push([x, y, part.centerZ + part.halfHeight]);
    }
    for (const ring of [bottom, top]) vertices.push(...ring);
    for (let i = 0; i < radial; i += 1) {
      const j = (i + 1) % radial;
      faces.push([base + i, base + j, base + radial + j]);
      faces.push([base + i, base + radial + j, base + radial + i]);
    }
  }
  return {
    vertices,
    faces,
    provenance: {
      source: 'in-source: src/body-conditioning.mjs#createBodyMesh',
      revision: 'generated',
      license: 'MIT',
    },
  };
}

/**
 * Rasterize a mesh into position, normal, and depth maps.
 *
 * Bounded: per-pixel edge test with a z-buffer; cost is O(width * height
 * * faces) with a small constant, so the benchmark stays finite.
 *
 * @param {object} mesh - { vertices, faces } from createBodyMesh.
 * @param {object} [options]
 * @param {number} [options.width=64]
 * @param {number} [options.height=64]
 * @returns {{ width, height, position: Float32Array, normal: Float32Array, depth: Float32Array, pixels: number }}
 */
export function rasterizeBodyMesh(mesh, options = {}) {
  const width = Math.max(1, Math.floor(options.width ?? 64));
  const height = Math.max(1, Math.floor(options.height ?? 64));
  const position = new Float32Array(width * height * 3);
  const normal = new Float32Array(width * height * 3);
  const depth = new Float32Array(width * height).fill(Number.POSITIVE_INFINITY);
  let pixels = 0;
  const verts = mesh.vertices;
  const faces = mesh.faces;
  for (const face of faces) {
    const a = verts[face[0]];
    const b = verts[face[1]];
    const c = verts[face[2]];
    const ux = b[0] - a[0];
    const uy = b[1] - a[1];
    const uz = b[2] - a[2];
    const vx = c[0] - a[0];
    const vy = c[1] - a[1];
    const vz = c[2] - a[2];
    let nx = uy * vz - uz * vy;
    let ny = uz * vx - ux * vz;
    let nz = ux * vy - uy * vx;
    const nLen = Math.hypot(nx, ny, nz);
    if (nLen === 0) continue;
    nx /= nLen;
    ny /= nLen;
    nz /= nLen;
    const ax = Math.max(0, Math.min(width - 1, Math.round((a[0] + 1) * 0.5 * (width - 1))));
    const ay = Math.max(0, Math.min(height - 1, Math.round((a[1] + 1) * 0.5 * (height - 1))));
    const bx = Math.max(0, Math.min(width - 1, Math.round((b[0] + 1) * 0.5 * (width - 1))));
    const by = Math.max(0, Math.min(height - 1, Math.round((b[1] + 1) * 0.5 * (height - 1))));
    const cx = Math.max(0, Math.min(width - 1, Math.round((c[0] + 1) * 0.5 * (width - 1))));
    const cy = Math.max(0, Math.min(height - 1, Math.round((c[1] + 1) * 0.5 * (height - 1))));
    const minX = Math.min(ax, bx, cx);
    const maxX = Math.max(ax, bx, cx);
    const minY = Math.min(ay, by, cy);
    const maxY = Math.max(ay, by, cy);
    const edge = (p0x, p0y, p1x, p1y, x, y) =>
      (x - p0x) * (p1y - p0y) - (y - p0y) * (p1x - p0x);
    for (let y = minY; y <= maxY; y += 1) {
      for (let x = minX; x <= maxX; x += 1) {
        const w0 = edge(bx, by, cx, cy, x, y);
        const w1 = edge(cx, cy, ax, ay, x, y);
        const w2 = edge(ax, ay, bx, by, x, y);
        const hasNeg = w0 < 0 || w1 < 0 || w2 < 0;
        const hasPos = w0 > 0 || w1 > 0 || w2 > 0;
        if (hasNeg && hasPos) continue;
        const z = (a[2] + b[2] + c[2]) / 3;
        const idx = y * width + x;
        if (z < depth[idx]) {
          depth[idx] = z;
          position[idx * 3 + 0] = (a[0] + b[0] + c[0]) / 3;
          position[idx * 3 + 1] = (a[1] + b[1] + c[1]) / 3;
          position[idx * 3 + 2] = z;
          normal[idx * 3 + 0] = nx;
          normal[idx * 3 + 1] = ny;
          normal[idx * 3 + 2] = nz;
          pixels += 1;
        }
      }
    }
  }
  return { width, height, position, normal, depth, pixels };
}

/**
 * The DenseBodyConditioning capsule: the only place that knows how the dense
 * representation is produced. The renderer consumes the three maps only.
 */
export class DenseBodyConditioning {
  constructor(mesh, options = {}) {
    this.mesh = mesh;
    this.width = options.width ?? 64;
    this.height = options.height ?? 64;
    if (mesh.provenance) verifyProvenance(mesh.provenance);
  }

  /**
   * @returns {{ position: Float32Array, normal: Float32Array, depth: Float32Array, width: number, height: number }}
   */
  rasterize() {
    const result = rasterizeBodyMesh(this.mesh, {
      width: this.width,
      height: this.height,
    });
    return {
      width: result.width,
      height: result.height,
      position: result.position,
      normal: result.normal,
      depth: result.depth,
    };
  }
}

