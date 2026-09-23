import { useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { ContactShadows, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { decodeHeights, type JobResult } from "../api";

export type Finish = "wood" | "terrain";

type Props = {
  result: JobResult;
  mode: "stepped" | "smooth";
  finish: Finish;
  vertical: number; // vertical exaggeration
  carve: number; // number of depth levels revealed (Infinity = all)
  autoRotate: boolean;
};

const SURROUND = new THREE.Color("#2b3038");

function hash(x: number, y: number) {
  const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return s - Math.floor(s);
}

/** Top surface + skirt walls + base, all from one height grid. */
function buildGeometry(w: number, h: number, W: number, H: number) {
  const nTop = w * h;
  const ring: number[] = [];
  // Perimeter, counter-clockwise seen from above (+y): near edge (+z) first.
  for (let x = 0; x < w; x++) ring.push((h - 1) * w + x);
  for (let y = h - 2; y >= 0; y--) ring.push(y * w + (w - 1));
  for (let x = w - 2; x >= 0; x--) ring.push(x);
  for (let y = 1; y < h - 1; y++) ring.push(y * w);
  const m = ring.length;
  const pos = new Float32Array((nTop + m) * 3);
  for (let j = 0; j < h; j++)
    for (let i = 0; i < w; i++) {
      const k = (j * w + i) * 3;
      pos[k] = (i / (w - 1) - 0.5) * W;
      pos[k + 2] = (j / (h - 1) - 0.5) * H; // three.js: y up, board rows along +z
    }
  for (let r = 0; r < m; r++) {
    const s = ring[r] * 3, d = (nTop + r) * 3;
    pos[d] = pos[s];
    pos[d + 2] = pos[s + 2];
  }
  const idx: number[] = [];
  for (let j = 0; j < h - 1; j++)
    for (let i = 0; i < w - 1; i++) {
      const a = j * w + i, b = a + 1, c = a + w, d = c + 1;
      idx.push(a, c, b, b, c, d);
    }
  for (let r = 0; r < m; r++) {
    const t0 = ring[r], t1 = ring[(r + 1) % m], b0 = nTop + r, b1 = nTop + ((r + 1) % m);
    idx.push(t0, b1, t1, t0, b0, b1);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.BufferAttribute(new Float32Array((nTop + m) * 3), 3));
  g.setIndex(idx);
  return { g, ring, nTop };
}

function Relief({ result, mode, finish, vertical, carve }: Omit<Props, "autoRotate">) {
  const { stats, heightmap } = result;
  const [W, H, T] = stats.board_in;
  const w = heightmap.width, h = heightmap.height;

  const heights = useMemo(
    () => decodeHeights(mode === "stepped" ? heightmap.stepped : heightmap.smooth, T),
    [heightmap, mode, T],
  );
  const geo = useMemo(() => buildGeometry(w, h, W, H), [w, h, W, H]);
  useEffect(() => () => geo.g.dispose(), [geo]);

  // Distinct cut depths in order: layer depths, then the surround.
  const levels = useMemo(() => {
    const d = stats.layers.map((l) => l.depth_in);
    if (stats.surround_depth_in != null) d.push(stats.surround_depth_in);
    return d;
  }, [stats]);

  // Grow-in animation whenever the result changes.
  const grow = useRef(0);
  useEffect(() => { grow.current = 0; }, [result]);

  const apply = (g: number) => {
    const pos = geo.g.attributes.position as THREE.BufferAttribute;
    const col = geo.g.attributes.color as THREE.BufferAttribute;
    const n = w * h;
    const maxCut = carve >= levels.length ? Infinity : levels[Math.max(0, carve)] ?? 0;
    const layerCols = stats.layers.map((l) => new THREE.Color(l.color));
    const wood = new THREE.Color("#c99a62"), woodDark = new THREE.Color("#7a5230"), tmp = new THREE.Color();
    const floor = T - (stats.surround_depth_in ?? levels[levels.length - 1]);
    for (let k = 0; k < n; k++) {
      const real = heights[k];
      const depth = Math.min(T - real, maxCut);
      const z = T - depth;
      pos.setY(k, (z - T) * vertical * g + T * vertical);
      // colour
      const x = pos.getX(k), zz = pos.getZ(k);
      if (finish === "wood") {
        const grain = 0.5 + 0.5 * Math.sin(zz * 9 + Math.sin(x * 1.3) * 2.2 + hash(Math.round(x * 30), 0) * 0.6);
        const cut = (T - z) / Math.max(1e-6, T - floor);
        tmp.copy(wood).lerp(woodDark, 0.15 + grain * 0.25 + cut * 0.35);
      } else if (stats.surround_depth_in != null && depth >= stats.surround_depth_in - 1e-4) {
        tmp.copy(SURROUND);
      } else {
        // nearest layer by depth; interpolate for the smooth relief
        const step = levels.length > 1 ? levels[1] - levels[0] : 1;
        const f = Math.max(0, Math.min(layerCols.length - 1, (T - real) / step));
        const i0 = Math.floor(f), i1 = Math.min(layerCols.length - 1, i0 + 1);
        tmp.copy(layerCols[i0]).lerp(layerCols[i1], mode === "smooth" ? f - i0 : 0);
        if (depth < T - real - 1e-4) tmp.lerp(wood, 0.85); // not yet carved: raw wood
      }
      col.setXYZ(k, tmp.r, tmp.g, tmp.b);
    }
    // Skirt: base vertices at y=0, shaded like the board edge.
    const edge = new THREE.Color(finish === "wood" ? "#8a6038" : "#1e232b");
    for (let r = 0; r < geo.ring.length; r++) {
      pos.setY(geo.nTop + r, 0);
      col.setXYZ(geo.nTop + r, edge.r, edge.g, edge.b);
    }
    pos.needsUpdate = true;
    col.needsUpdate = true;
    geo.g.computeVertexNormals();
  };

  useEffect(() => { apply(grow.current); });

  useFrame((_, dt) => {
    if (grow.current < 1) {
      grow.current = Math.min(1, grow.current + dt * 1.4);
      const e = 1 - Math.pow(1 - grow.current, 3);
      apply(e);
    }
  });

  return (
    <mesh geometry={geo.g} castShadow receiveShadow>
      <meshStandardMaterial vertexColors roughness={finish === "wood" ? 0.72 : 0.55} metalness={0.02} />
    </mesh>
  );
}

/** Frame the board for the current viewport aspect whenever the board size changes. */
function CameraFit({ W, H }: { W: number; H: number }) {
  const { camera, size, controls } = useThree();
  useEffect(() => {
    const cam = camera as THREE.PerspectiveCamera;
    const aspect = size.width / Math.max(1, size.height);
    const vfov = (cam.fov * Math.PI) / 180;
    const hfov = 2 * Math.atan(Math.tan(vfov / 2) * aspect);
    // Distance so the board's diagonal fits in both directions, with some margin.
    const r = Math.hypot(W, H) / 2;
    const d = (r / Math.sin(Math.min(vfov, hfov) / 2)) * 0.95;
    const dir = new THREE.Vector3(0, 0.72, 0.7).normalize();
    cam.position.copy(dir.multiplyScalar(d));
    cam.lookAt(0, 0, 0);
    (controls as unknown as { target?: THREE.Vector3; update?: () => void })?.target?.set(0, 0, 0);
    (controls as unknown as { update?: () => void })?.update?.();
  }, [W, H, size.width, size.height, camera, controls]);
  return null;
}

export default function ReliefView(props: Props) {
  const [W, H] = props.result.stats.board_in;
  const span = Math.max(W, H);
  return (
    <Canvas shadows dpr={[1, 2]} camera={{ position: [0, span * 0.85, span * 0.95], fov: 40 }}>
      <color attach="background" args={["#0b0e13"]} />
      <fog attach="fog" args={["#0b0e13", span * 3, span * 6]} />
      <hemisphereLight args={["#cfe9ff", "#1a1410", 0.55]} />
      <directionalLight
        position={[-span, span * 1.2, -span * 0.4]}
        intensity={2.4}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-left={-span}
        shadow-camera-right={span}
        shadow-camera-top={span}
        shadow-camera-bottom={-span}
      />
      <directionalLight position={[span, span * 0.4, span]} intensity={0.5} color="#7fe3d2" />
      <CameraFit W={W} H={H} />
      <Relief {...props} />
      <ContactShadows position={[0, -0.01, 0]} opacity={0.6} scale={span * 2.2} blur={2.6} far={span} />
      <OrbitControls
        makeDefault
        enableDamping
        autoRotate={props.autoRotate}
        autoRotateSpeed={0.6}
        maxPolarAngle={Math.PI * 0.48}
        minDistance={span * 0.3}
        maxDistance={span * 4}
      />
    </Canvas>
  );
}
