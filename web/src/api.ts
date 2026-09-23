export type BoardSpec = {
  width_in: number;
  height_in: number;
  thickness_in: number;
  padding_in: number;
  floor_in: number;
  resolution_in: number;
};

export type LayerSpec = {
  count: number;
  max_depth_in: number | null;
  smoothing_in: number;
  curve: "linear" | "equal-area";
  exaggeration: number;
  min_feature_in2: number;
  surround: "raised" | "frame";
};

export type Region =
  | { kind: "state"; usps: string; name: string }
  | { kind: "custom"; geojson: GeoJSON.Polygon; label: string };

export type LayerStat = {
  index: number;
  depth_in: number;
  elev_m: [number, number];
  area_in2: number;
  color: string;
};

export type Stats = {
  name: string;
  elevation_m: [number, number];
  map_scale: number;
  ground_px_m: number;
  grid: [number, number];
  board_in: [number, number, number];
  surround_depth_in: number | null;
  timings_s: Record<string, number>;
  layers: LayerStat[];
};

export type JobResult = {
  stats: Stats;
  heightmap: { width: number; height: number; stepped: string; smooth: string };
  svg: { layers: string[]; surround: string; outline: string };
};

export type JobState = {
  id: string;
  status: "queued" | "running" | "done" | "error" | "cancelled";
  message: string;
  progress: number;
  error: string | null;
  result?: JobResult;
};

export const DEFAULT_BOARD: BoardSpec = {
  width_in: 11,
  height_in: 8,
  thickness_in: 0.75,
  padding_in: 0.25,
  floor_in: 0.25,
  resolution_in: 0.01,
};

export const DEFAULT_LAYERS: LayerSpec = {
  count: 8,
  max_depth_in: null,
  smoothing_in: 0.04,
  curve: "linear",
  exaggeration: 1,
  min_feature_in2: 0.005,
  surround: "raised",
};

/**
 * API URL relative to the page, never "/api/...": the app is served at "/" and under "/topology/",
 * and a relative URL resolves against whichever base the page was loaded from.
 */
export const apiUrl = (path: string) => `api/${path}`;

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const body = await r.json();
      if (body?.detail) msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
}

export function fetchStates(): Promise<GeoJSON.FeatureCollection> {
  return fetch(apiUrl("states")).then((r) => json<GeoJSON.FeatureCollection>(r));
}

/** Submit a job and poll until it finishes. Resolves with the final state; rejects on abort. */
export async function runJob(
  region: Region,
  board: BoardSpec,
  layers: LayerSpec,
  onUpdate: (s: JobState) => void,
  signal: AbortSignal,
): Promise<JobState> {
  const body =
    region.kind === "state"
      ? { state: region.usps, board, layers }
      : { geojson: region.geojson, name: region.label, board, layers };
  let s = await json<JobState>(
    await fetch(apiUrl("jobs"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    }),
  );
  onUpdate(s);
  const id = s.id;
  // Tell the server to drop the job if we stop caring before it starts.
  signal.addEventListener("abort", () => { fetch(apiUrl(`jobs/${id}`), { method: "DELETE" }).catch(() => {}); }, { once: true });
  let delay = 150;
  while (s.status === "queued" || s.status === "running") {
    if (signal.aborted) throw new DOMException("aborted", "AbortError");
    await new Promise((res) => setTimeout(res, delay));
    delay = Math.min(600, delay * 1.3);
    if (signal.aborted) throw new DOMException("aborted", "AbortError");
    s = await json<JobState>(await fetch(apiUrl(`jobs/${s.id}`), { signal }));
    onUpdate(s);
  }
  return s;
}

export function decodeHeights(b64: string, thickness: number): Float32Array {
  const bin = atob(b64);
  const u16 = new Uint16Array(bin.length / 2);
  for (let i = 0; i < u16.length; i++) u16[i] = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = (u16[i] / 65535) * thickness;
  return out;
}

export const fmtIn = (v: number, digits = 3) => `${v.toFixed(digits)}″`;
export const fmtM = (v: number) => `${Math.round(v).toLocaleString()} m`;
