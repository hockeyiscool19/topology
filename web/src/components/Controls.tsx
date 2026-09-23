import { useMemo, useState, type ReactNode } from "react";
import type { BoardSpec, LayerSpec, Region } from "../api";
import type { DrawMode } from "./MapView";

type Props = {
  region: Region | null;
  states: { usps: string; name: string }[];
  onRegion: (r: Region) => void;
  drawMode: DrawMode;
  onDrawMode: (m: DrawMode) => void;
  board: BoardSpec;
  onBoard: (b: BoardSpec) => void;
  layers: LayerSpec;
  onLayers: (l: LayerSpec) => void;
};

function Section({ title, icon, children, defaultOpen = true }: { title: string; icon: ReactNode; children: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className={"section" + (open ? " open" : "")}>
      <button className="section-head" onClick={() => setOpen(!open)}>
        <span className="section-icon">{icon}</span>
        {title}
        <span className="chev">▾</span>
      </button>
      <div className="section-body">{children}</div>
    </section>
  );
}

function Num({ label, value, onChange, step = 0.25, min = 0, unit = "in", hint }: {
  label: string; value: number; onChange: (v: number) => void; step?: number; min?: number; unit?: string; hint?: string;
}) {
  const [text, setText] = useState<string | null>(null);
  return (
    <label className="field" title={hint}>
      <span className="field-label">{label}</span>
      <span className="num">
        <input
          type="number"
          step={step}
          min={min}
          value={text ?? String(value)}
          onChange={(e) => {
            setText(e.target.value);
            const v = parseFloat(e.target.value);
            if (Number.isFinite(v) && v >= min) onChange(v);
          }}
          onBlur={() => setText(null)}
        />
        <em>{unit}</em>
      </span>
    </label>
  );
}

function Slider({ label, value, min, max, step, onChange, format, hint }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; format: (v: number) => string; hint?: string;
}) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <label className="slider" title={hint}>
      <span className="slider-top">
        <span className="field-label">{label}</span>
        <span className="mono val">{format(value)}</span>
      </span>
      <input
        type="range" min={min} max={max} step={step} value={value}
        style={{ ["--pct" as string]: `${pct}%` }}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </label>
  );
}

function Seg<T extends string>({ value, options, onChange }: { value: T; options: { v: T; label: string; hint?: string }[]; onChange: (v: T) => void }) {
  const i = options.findIndex((o) => o.v === value);
  return (
    <div className="seg" style={{ ["--n" as string]: options.length, ["--i" as string]: i }}>
      <span className="seg-thumb" />
      {options.map((o) => (
        <button key={o.v} className={o.v === value ? "on" : ""} title={o.hint} onClick={() => onChange(o.v)}>{o.label}</button>
      ))}
    </div>
  );
}

const I = {
  pin: <svg viewBox="0 0 24 24"><path d="M12 22s7-6.2 7-12a7 7 0 1 0-14 0c0 5.8 7 12 7 12z" /><circle cx="12" cy="10" r="2.5" /></svg>,
  board: <svg viewBox="0 0 24 24"><path d="M3 8l9-5 9 5-9 5-9-5z" /><path d="M3 8v6l9 5 9-5V8" /></svg>,
  layers: <svg viewBox="0 0 24 24"><path d="M12 3l9 5-9 5-9-5 9-5z" /><path d="M3 13l9 5 9-5" /><path d="M3 17.5l9 5 9-5" opacity=".5" /></svg>,
  tune: <svg viewBox="0 0 24 24"><path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0" /><circle cx="16" cy="6" r="2" /><circle cx="10" cy="12" r="2" /><circle cx="18" cy="18" r="2" /></svg>,
};

export default function Controls(p: Props) {
  const { board, layers } = p;
  const [query, setQuery] = useState("");
  const maxCut = +(board.thickness_in - board.floor_in).toFixed(4);
  const depth = layers.max_depth_in ?? maxCut;
  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    return p.states.filter((s) => s.name.toLowerCase().includes(q) || s.usps.toLowerCase() === q).slice(0, 6);
  }, [query, p.states]);
  const step = depth / (layers.surround === "raised" ? layers.count : layers.count - 1);

  const setB = (k: keyof BoardSpec) => (v: number) => {
    const next = { ...board, [k]: v };
    p.onBoard(next);
    // A custom max depth that no longer fits the board falls back to "full cuttable depth".
    if (layers.max_depth_in != null && layers.max_depth_in > next.thickness_in - next.floor_in) {
      p.onLayers({ ...layers, max_depth_in: null });
    }
  };
  const setL = <K extends keyof LayerSpec>(k: K) => (v: LayerSpec[K]) => p.onLayers({ ...layers, [k]: v });

  return (
    <div className="controls">
      <Section title="Region" icon={I.pin}>
        <div className="search">
          <input
            placeholder="Search a state…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && matches[0]) {
                p.onRegion({ kind: "state", ...matches[0] });
                setQuery("");
              }
            }}
          />
          {matches.length > 0 && (
            <div className="search-results glass">
              {matches.map((s) => (
                <button key={s.usps} onClick={() => { p.onRegion({ kind: "state", ...s }); setQuery(""); }}>
                  <b>{s.usps}</b> {s.name}
                </button>
              ))}
            </div>
          )}
        </div>
        <div className="draw-row">
          <button className={"chip" + (p.drawMode === "rect" ? " on" : "")} onClick={() => p.onDrawMode(p.drawMode === "rect" ? "none" : "rect")}>
            ▭ Draw rectangle
          </button>
          <button className={"chip" + (p.drawMode === "poly" ? " on" : "")} onClick={() => p.onDrawMode(p.drawMode === "poly" ? "none" : "poly")}>
            ⬠ Draw polygon
          </button>
        </div>
        <div className="region-pill">
          {p.region ? (
            <>
              <span className="dot" />
              {p.region.kind === "state" ? p.region.name : "Custom area"}
              {p.region.kind === "state" && <em>{p.region.usps}</em>}
            </>
          ) : (
            <span className="dim">Click a state on the map, search, or draw any area.</span>
          )}
        </div>
      </Section>

      <Section title="Board" icon={I.board}>
        <div className="grid2">
          <Num label="Width" value={board.width_in} onChange={setB("width_in")} min={1} />
          <Num label="Height" value={board.height_in} onChange={setB("height_in")} min={1} />
          <Num label="Thickness" value={board.thickness_in} onChange={setB("thickness_in")} step={0.125} min={0.1} />
          <Num label="Padding" value={board.padding_in} onChange={setB("padding_in")} step={0.125} />
          <Num label="Floor" value={board.floor_in} onChange={setB("floor_in")} step={0.0625} hint="Material left uncut under the deepest pocket" />
          <Num label="Max depth" value={+depth.toFixed(4)} onChange={(v) => setL("max_depth_in")(v >= maxCut ? null : v)} step={0.0625} min={0.01} hint="Deepest cut (defaults to thickness − floor)" />
        </div>
      </Section>

      <Section title="Layers" icon={I.layers}>
        <Slider label="Layers" value={layers.count} min={2} max={24} step={1} onChange={setL("count")}
                format={(v) => `${v} · ${step.toFixed(3)}″ step`} hint="Distinct heights, including the uncut top" />
        <Seg
          value={layers.curve}
          onChange={setL("curve")}
          options={[
            { v: "linear", label: "Equal height", hint: "Each layer spans the same elevation range" },
            { v: "equal-area", label: "Equal area", hint: "Each layer covers a similar share of the map (great for flat states)" },
          ]}
        />
        <Slider label="Peak emphasis" value={layers.exaggeration} min={0.4} max={3} step={0.05} onChange={setL("exaggeration")}
                format={(v) => (Math.abs(v - 1) < 0.01 ? "neutral" : v > 1 ? `peaks ×${v.toFixed(2)}` : `lowlands ×${(1 / v).toFixed(2)}`)}
                hint="Spend more layers on high ground (right) or low ground (left)" />
        <Seg
          value={layers.surround}
          onChange={setL("surround")}
          options={[
            { v: "raised", label: "Raised region", hint: "Cut away everything outside the region" },
            { v: "frame", label: "Inset in frame", hint: "Leave the board around the region at full height" },
          ]}
        />
      </Section>

      <Section title="Detail" icon={I.tune} defaultOpen={false}>
        <Slider label="Smoothing" value={layers.smoothing_in} min={0} max={0.25} step={0.005} onChange={setL("smoothing_in")}
                format={(v) => (v === 0 ? "off" : `${v.toFixed(3)}″`)} hint="Terrain blur radius on the board" />
        <Slider label="Min feature" value={layers.min_feature_in2} min={0} max={0.1} step={0.001} onChange={setL("min_feature_in2")}
                format={(v) => `${v.toFixed(3)} in²`} hint="Drop islands and holes smaller than this — keep it above your bit's footprint" />
        <Slider label="Resolution" value={board.resolution_in} min={0.005} max={0.05} step={0.001} onChange={setB("resolution_in")}
                format={(v) => `${v.toFixed(3)}″/px`} hint="Raster grid on the board. Smaller = finer and slower" />
      </Section>
    </div>
  );
}
