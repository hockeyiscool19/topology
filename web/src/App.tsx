import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import {
  apiUrl, DEFAULT_BOARD, DEFAULT_LAYERS, fetchStates, runJob,
  type BoardSpec, type JobState, type LayerSpec, type Region,
} from "./api";
import Controls from "./components/Controls";
import MapView, { type DrawMode } from "./components/MapView";
import LayersView from "./components/LayersView";
import type { Finish } from "./components/ReliefView";

const ReliefView = lazy(() => import("./components/ReliefView"));

type Tab = "map" | "layers" | "3d";

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

function Logo() {
  return (
    <svg viewBox="0 0 40 40" className="logo">
      <defs>
        <linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#6ef3da" />
          <stop offset="1" stopColor="#f7b955" />
        </linearGradient>
      </defs>
      {[0, 1, 2, 3].map((i) => (
        <path key={i} d={`M${4 + i * 2} ${32 - i * 5} Q20 ${6 + i * 3} ${36 - i * 2} ${32 - i * 5}`}
              fill="none" stroke="url(#lg)" strokeWidth="2" opacity={1 - i * 0.18} />
      ))}
    </svg>
  );
}

export default function App() {
  const [region, setRegion] = useState<Region | null>(null);
  const [board, setBoard] = useState<BoardSpec>(DEFAULT_BOARD);
  const [layers, setLayers] = useState<LayerSpec>(DEFAULT_LAYERS);
  const [drawMode, setDrawMode] = useState<DrawMode>("none");
  const [tab, setTab] = useState<Tab>("map");
  const [job, setJob] = useState<JobState | null>(null);
  const [done, setDone] = useState<JobState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [states, setStates] = useState<{ usps: string; name: string }[]>([]);
  const [terrain3d, setTerrain3d] = useState(false);

  // 3D view options
  const [mode, setMode] = useState<"stepped" | "smooth">("stepped");
  const [finish, setFinish] = useState<Finish>("terrain");
  const [vertical, setVertical] = useState(2.5);
  const [carve, setCarve] = useState(99);
  const [playing, setPlaying] = useState(false);
  const [autoRotate, setAutoRotate] = useState(true);

  useEffect(() => {
    fetchStates()
      .then((fc) => setStates(
        fc.features.map((f) => ({ usps: f.properties!.usps, name: f.properties!.name }))
          .sort((a, b) => a.name.localeCompare(b.name)),
      ))
      .catch(() => setError("Can't reach the Topology API. Is `topology serve` running?"));
  }, []);

  // Regenerate whenever inputs settle.
  const params = useDebounced(useMemo(() => ({ region, board, layers }), [region, board, layers]), 450);
  const firstForRegion = useRef<Region | null>(null);
  useEffect(() => {
    if (!params.region) return;
    const ctl = new AbortController();
    setError(null);
    runJob(params.region, params.board, params.layers, setJob, ctl.signal)
      .then((s) => {
        if (s.status === "error") { setError(s.error ?? "Generation failed"); return; }
        setDone(s);
        setCarve(99);
        if (firstForRegion.current !== params.region) {
          firstForRegion.current = params.region;
          setTab((t) => (t === "map" ? "3d" : t));
        }
      })
      .catch((e) => { if (e.name !== "AbortError") setError(String(e.message ?? e)); });
    return () => ctl.abort();
  }, [params]);

  const result = done?.result;
  const levels = result ? result.stats.layers.length + (result.stats.surround_depth_in != null ? 1 : 0) : 0;
  const busy = job != null && (job.status === "queued" || job.status === "running");

  // Carve playback
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setCarve((c) => {
        if (c >= levels) { setPlaying(false); return levels; }
        return c + 1;
      });
    }, 650);
    return () => clearInterval(id);
  }, [playing, levels]);

  const s = result?.stats;

  return (
    <div className="app">
      <div className="bg-contours" />
      <header className="topbar glass">
        <div className="brand">
          <Logo />
          <div>
            <h1>Topology</h1>
            <p>Terrain → CNC carving files</p>
          </div>
        </div>
        <nav className="tabs">
          {([["map", "Map"], ["layers", "Toolpaths"], ["3d", "3D Preview"]] as [Tab, string][]).map(([t, label]) => (
            <button key={t} className={tab === t ? "on" : ""} onClick={() => setTab(t)} disabled={t !== "map" && !result}>
              {label}
            </button>
          ))}
        </nav>
        <a
          className={"btn-primary" + (result && !busy ? "" : " disabled")}
          href={done ? apiUrl(`jobs/${done.id}/download`) : undefined}
          download
        >
          <svg viewBox="0 0 24 24"><path d="M12 3v12m0 0l-5-5m5 5l5-5M4 21h16" /></svg>
          Download SVG + STL
        </a>
      </header>

      <aside className="panel glass">
        <Controls
          region={region}
          states={states}
          onRegion={(r) => { setRegion(r); }}
          drawMode={drawMode}
          onDrawMode={(m) => { setDrawMode(m); setTab("map"); }}
          board={board}
          onBoard={setBoard}
          layers={layers}
          onLayers={setLayers}
        />
        <div className="status">
          {error ? (
            <div className="error">⚠ {error}</div>
          ) : busy ? (
            <>
              <div className="status-row"><span className="spinner" />{job!.message}<span className="mono dim">{Math.round(job!.progress * 100)}%</span></div>
              <div className="progress"><i style={{ width: `${Math.max(4, job!.progress * 100)}%` }} /></div>
            </>
          ) : s ? (
            <div className="status-row ok">✓ {s.layers.length} layers in {Object.values(s.timings_s).reduce((a, b) => a + b, 0).toFixed(1)}s</div>
          ) : (
            <div className="status-row dim">Pick a region to begin — previews update live.</div>
          )}
        </div>
      </aside>

      <main className="stage">
        <div className={"view" + (tab === "map" ? " show" : "")}>
          <MapView region={region} onRegion={setRegion} drawMode={drawMode}
                   onDrawDone={() => setDrawMode("none")} terrain3d={terrain3d} />
          <div className="map-tools">
            <button className={"chip glass" + (terrain3d ? " on" : "")} onClick={() => setTerrain3d(!terrain3d)}>
              {terrain3d ? "◆ 3D terrain on" : "◇ 3D terrain"}
            </button>
          </div>
          {!region && (
            <div className="hero">
              <h2>Carve any landscape.</h2>
              <p>Select a state or draw an area. Topology slices real elevation data into
                 clean, non-overlapping pocket layers for your CNC — plus a watertight STL.</p>
            </div>
          )}
        </div>

        {result && tab === "layers" && (
          <div className="view show"><LayersView result={result} /></div>
        )}

        {result && tab === "3d" && (
          <div className="view show">
            <Suspense fallback={<div className="center-msg"><span className="spinner" /> Loading 3D…</div>}>
              <ReliefView result={result} mode={mode} finish={finish} vertical={vertical}
                          carve={Math.min(carve, levels)} autoRotate={autoRotate && !playing} />
            </Suspense>
            <div className="toolbar glass">
              <div className="seg small" style={{ ["--n" as string]: 2, ["--i" as string]: mode === "stepped" ? 0 : 1 }}>
                <span className="seg-thumb" />
                <button className={mode === "stepped" ? "on" : ""} onClick={() => setMode("stepped")}>Stepped</button>
                <button className={mode === "smooth" ? "on" : ""} onClick={() => setMode("smooth")}>Smooth</button>
              </div>
              <div className="seg small" style={{ ["--n" as string]: 2, ["--i" as string]: finish === "terrain" ? 0 : 1 }}>
                <span className="seg-thumb" />
                <button className={finish === "terrain" ? "on" : ""} onClick={() => setFinish("terrain")}>Terrain</button>
                <button className={finish === "wood" ? "on" : ""} onClick={() => setFinish("wood")}>Wood</button>
              </div>
              <label className="mini-slider">
                <span>Vertical ×{vertical.toFixed(1)}</span>
                <input type="range" min={1} max={8} step={0.1} value={vertical}
                       style={{ ["--pct" as string]: `${((vertical - 1) / 7) * 100}%` }}
                       onChange={(e) => setVertical(parseFloat(e.target.value))} />
              </label>
              <label className="mini-slider">
                <span>Carve pass {Math.min(carve, levels)}/{levels}</span>
                <input type="range" min={0} max={levels} step={1} value={Math.min(carve, levels)}
                       style={{ ["--pct" as string]: `${(Math.min(carve, levels) / Math.max(1, levels)) * 100}%` }}
                       onChange={(e) => { setPlaying(false); setCarve(parseInt(e.target.value)); }} />
              </label>
              <button className="icon-btn" title="Simulate carving"
                      onClick={() => { if (!playing) setCarve(0); setPlaying(!playing); }}>
                {playing ? "❚❚" : "▶"}
              </button>
              <button className={"icon-btn" + (autoRotate ? " on" : "")} title="Auto-rotate" onClick={() => setAutoRotate(!autoRotate)}>⟳</button>
            </div>
          </div>
        )}

        {busy && <div className="stage-progress"><i style={{ width: `${job!.progress * 100}%` }} /></div>}

        {s && tab !== "map" && (
          <div className="statbar glass">
            <div><em>Region</em><b>{region?.kind === "state" ? region.name : "Custom area"}</b></div>
            <div><em>Elevation</em><b className="mono">{Math.round(s.elevation_m[0]).toLocaleString()}–{Math.round(s.elevation_m[1]).toLocaleString()} m</b></div>
            <div><em>Scale</em><b className="mono">1:{s.map_scale.toLocaleString()}</b></div>
            <div><em>Relief</em><b className="mono">{(s.elevation_m[1] - s.elevation_m[0]).toFixed(0)} m → {(s.surround_depth_in ?? s.layers[s.layers.length - 1].depth_in).toFixed(3)}″</b></div>
            <div><em>Grid</em><b className="mono">{s.grid[1]}×{s.grid[0]}</b></div>
          </div>
        )}
      </main>
    </div>
  );
}
