import { useEffect, useRef, useState } from "react";
import * as maplibregl from "maplibre-gl";
import type { GeoJSONSource, MapMouseEvent } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
// maplibre v6 runs tiles in an ES-module worker; let Vite bundle it and hand over the URL.
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import { fetchStates, type Region } from "../api";

maplibregl.setWorkerUrl(workerUrl);

export type DrawMode = "none" | "rect" | "poly";

const TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png";
// Keyless vector basemap (OpenStreetMap data via OpenFreeMap).
const STYLE_URL = "https://tiles.openfreemap.org/styles/dark";

function addTerrain(map: maplibregl.Map) {
  const dem = (): maplibregl.RasterDEMSourceSpecification => ({
    type: "raster-dem", tiles: [TERRARIUM], encoding: "terrarium", tileSize: 256, maxzoom: 14,
    attribution: "Terrain: AWS Terrain Tiles",
  });
  map.addSource("hills", dem());
  map.addSource("terrain", dem());
  // Hillshade goes under the first label layer so place names stay crisp.
  const firstSymbol = map.getStyle().layers.find((l) => l.type === "symbol")?.id;
  map.addLayer({
    id: "hillshade", type: "hillshade", source: "hills",
    paint: {
      "hillshade-shadow-color": "#000000",
      "hillshade-highlight-color": "#7cf0da",
      "hillshade-accent-color": "#0b2a33",
      "hillshade-exaggeration": 0.6,
    },
  }, firstSymbol);
}

const EMPTY: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };

type Props = {
  region: Region | null;
  onRegion: (r: Region) => void;
  drawMode: DrawMode;
  onDrawDone: () => void;
  terrain3d: boolean;
};

function boundsOf(g: GeoJSON.Geometry): [number, number, number, number] {
  let [w, s, e, n] = [180, 90, -180, -90];
  const visit = (c: unknown): void => {
    if (Array.isArray(c) && typeof c[0] === "number") {
      const [x, y] = c as number[];
      w = Math.min(w, x); e = Math.max(e, x); s = Math.min(s, y); n = Math.max(n, y);
    } else if (Array.isArray(c)) c.forEach(visit);
  };
  if ("coordinates" in g) visit(g.coordinates);
  return [w, s, e, n];
}

export default function MapView({ region, onRegion, drawMode, onDrawDone, terrain3d }: Props) {
  const el = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const statesRef = useRef<GeoJSON.FeatureCollection | null>(null);
  const [ready, setReady] = useState(false);
  const [hover, setHover] = useState<string | null>(null);
  // Latest props for event handlers registered once.
  const live = useRef({ drawMode, onRegion, onDrawDone });
  live.current = { drawMode, onRegion, onDrawDone };
  // The click that ends a drag/draw must not also select the state underneath.
  const drawEndedAt = useRef(0);

  useEffect(() => {
    const map = new maplibregl.Map({
      container: el.current!,
      style: STYLE_URL,
      center: [-96, 38.5],
      zoom: 3.4,
      attributionControl: { compact: true },
      maxPitch: 75,
    });
    mapRef.current = map;
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "bottom-right");

    map.on("load", async () => {
      addTerrain(map);
      map.addSource("states", { type: "geojson", data: EMPTY, promoteId: "usps" });
      map.addSource("selection", { type: "geojson", data: EMPTY });
      map.addSource("draft", { type: "geojson", data: EMPTY });
      map.addLayer({
        id: "states-fill", type: "fill", source: "states",
        paint: {
          "fill-color": "#39d0b4",
          "fill-opacity": ["case", ["boolean", ["feature-state", "hover"], false], 0.14, 0.0],
        },
      });
      map.addLayer({
        id: "states-line", type: "line", source: "states",
        paint: { "line-color": "#8fb7c9", "line-opacity": 0.35, "line-width": 0.8 },
      });
      map.addLayer({
        id: "sel-fill", type: "fill", source: "selection",
        paint: { "fill-color": "#39d0b4", "fill-opacity": 0.12 },
      });
      map.addLayer({
        id: "sel-glow", type: "line", source: "selection",
        paint: { "line-color": "#39d0b4", "line-width": 8, "line-blur": 6, "line-opacity": 0.55 },
      });
      map.addLayer({
        id: "sel-line", type: "line", source: "selection",
        paint: { "line-color": "#b8fff1", "line-width": 1.6 },
      });
      map.addLayer({
        id: "draft-fill", type: "fill", source: "draft", filter: ["==", "$type", "Polygon"],
        paint: { "fill-color": "#f7b955", "fill-opacity": 0.15 },
      });
      map.addLayer({
        id: "draft-line", type: "line", source: "draft",
        paint: { "line-color": "#f7b955", "line-width": 2, "line-dasharray": [2, 1] },
      });
      map.addLayer({
        id: "draft-pts", type: "circle", source: "draft", filter: ["==", "$type", "Point"],
        paint: { "circle-radius": 4, "circle-color": "#f7b955", "circle-stroke-color": "#111", "circle-stroke-width": 1.5 },
      });

      try {
        statesRef.current = await fetchStates();
        (map.getSource("states") as GeoJSONSource).setData(statesRef.current);
      } catch (e) {
        console.error("Failed to load states", e);
      }
      setReady(true);
    });

    let hovered: string | number | undefined;
    map.on("mousemove", "states-fill", (e) => {
      if (live.current.drawMode !== "none") return;
      const f = e.features?.[0];
      if (!f) return;
      if (hovered !== undefined) map.setFeatureState({ source: "states", id: hovered }, { hover: false });
      hovered = f.id;
      map.setFeatureState({ source: "states", id: hovered! }, { hover: true });
      map.getCanvas().style.cursor = "pointer";
      setHover(String(f.properties?.name ?? ""));
    });
    map.on("mouseleave", "states-fill", () => {
      if (hovered !== undefined) map.setFeatureState({ source: "states", id: hovered }, { hover: false });
      hovered = undefined;
      map.getCanvas().style.cursor = "";
      setHover(null);
    });
    map.on("click", "states-fill", (e) => {
      if (live.current.drawMode !== "none" || performance.now() - drawEndedAt.current < 400) return;
      const p = e.features?.[0]?.properties;
      if (p) live.current.onRegion({ kind: "state", usps: p.usps, name: p.name });
    });

    return () => map.remove();
  }, []);

  // Selection outline + fly to it.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    let geom: GeoJSON.Geometry | null = null;
    if (region?.kind === "state") {
      geom = statesRef.current?.features.find((f) => f.properties?.usps === region.usps)?.geometry ?? null;
    } else if (region?.kind === "custom") {
      geom = region.geojson;
    }
    const src = map.getSource("selection") as GeoJSONSource;
    src.setData(geom ? { type: "Feature", properties: {}, geometry: geom } : EMPTY);
    if (geom) {
      const [w, s, e, n] = boundsOf(geom);
      // Alaska spans the antimeridian; don't zoom out to the whole world for it.
      if (e - w < 180) map.fitBounds([[w, s], [e, n]], { padding: 80, duration: 1400, maxZoom: 11 });
    }
  }, [region, ready]);

  // 3D terrain toggle.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    if (terrain3d) {
      map.setTerrain({ source: "terrain", exaggeration: 1.6 });
      map.easeTo({ pitch: 60, bearing: -15, duration: 1200 });
    } else {
      map.setTerrain(null);
      map.easeTo({ pitch: 0, bearing: 0, duration: 900 });
    }
  }, [terrain3d, ready]);

  // Drawing interactions.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready || drawMode === "none") return;
    const draft = map.getSource("draft") as GeoJSONSource;
    const canvas = map.getCanvas();
    canvas.style.cursor = "crosshair";
    map.doubleClickZoom.disable();
    if (map.getPitch() > 0) map.easeTo({ pitch: 0, bearing: 0, duration: 500 });

    const finish = (ring: number[][], label: string) => {
      drawEndedAt.current = performance.now();
      draft.setData(EMPTY);
      const closed = [...ring, ring[0]];
      live.current.onRegion({ kind: "custom", geojson: { type: "Polygon", coordinates: [closed] }, label });
      live.current.onDrawDone();
    };
    const pt = (e: MapMouseEvent) => [e.lngLat.lng, e.lngLat.lat];
    const cleanups: (() => void)[] = [];
    const on = <K extends keyof maplibregl.MapEventType>(type: K, fn: (e: maplibregl.MapEventType[K]) => void) => {
      map.on(type, fn);
      cleanups.push(() => map.off(type, fn));
    };

    if (drawMode === "rect") {
      map.dragPan.disable();
      let start: number[] | null = null;
      const rect = (a: number[], b: number[]) => [a, [b[0], a[1]], b, [a[0], b[1]]];
      on("mousedown", (e) => { start = pt(e); });
      on("mousemove", (e) => {
        if (!start) return;
        const r = rect(start, pt(e));
        draft.setData({ type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [[...r, r[0]]] } });
      });
      on("mouseup", (e) => {
        if (!start) return;
        const a = start, b = pt(e);
        start = null;
        const pa = map.project(a as [number, number]), pb = map.project(b as [number, number]);
        if (Math.abs(pa.x - pb.x) < 8 || Math.abs(pa.y - pb.y) < 8) { draft.setData(EMPTY); return; }
        finish(rect(a, b), "custom area");
      });
      cleanups.push(() => map.dragPan.enable());
    } else {
      const pts: number[][] = [];
      let cursor: number[] | null = null;
      const render = () => {
        const line = cursor ? [...pts, cursor] : pts;
        const feats: GeoJSON.Feature[] = pts.map((p) => ({ type: "Feature", properties: {}, geometry: { type: "Point", coordinates: p } }));
        if (line.length >= 3) feats.push({ type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [[...line, line[0]]] } });
        else if (line.length === 2) feats.push({ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: line } });
        draft.setData({ type: "FeatureCollection", features: feats });
      };
      const close = () => { if (pts.length >= 3) finish(pts.slice(), "custom area"); };
      on("click", (e) => {
        if (pts.length >= 3) {
          const first = map.project(pts[0] as [number, number]);
          if (Math.hypot(first.x - e.point.x, first.y - e.point.y) < 10) return close();
        }
        pts.push(pt(e));
        render();
      });
      on("dblclick", (e) => { e.preventDefault(); pts.pop(); close(); });
      on("mousemove", (e) => { cursor = pt(e); render(); });
    }

    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === "Escape") { draft.setData(EMPTY); live.current.onDrawDone(); }
    };
    window.addEventListener("keydown", onKey);
    return () => {
      cleanups.forEach((c) => c());
      window.removeEventListener("keydown", onKey);
      canvas.style.cursor = "";
      map.doubleClickZoom.enable();
      draft.setData(EMPTY);
    };
  }, [drawMode, ready]);

  return (
    <div className="map-wrap">
      <div ref={el} className="map" />
      {drawMode !== "none" && (
        <div className="map-hint glass">
          {drawMode === "rect"
            ? "Drag to draw a rectangle"
            : "Click to add points · click the first point or double-click to finish"}
          <kbd>Esc</kbd>
        </div>
      )}
      {drawMode === "none" && hover && <div className="map-hint glass subtle">{hover} — click to select</div>}
    </div>
  );
}
