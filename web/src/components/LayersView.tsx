import { useState } from "react";
import { fmtIn, fmtM, type JobResult } from "../api";

export default function LayersView({ result }: { result: JobResult }) {
  const { stats, svg } = result;
  const [W, H] = stats.board_in;
  const [hover, setHover] = useState<number | null>(null);
  const maxArea = Math.max(...stats.layers.map((l) => l.area_in2), 1e-6);

  return (
    <div className="layers-view">
      <div className="board-stage">
        <svg viewBox={`${-0.4} ${-0.4} ${W + 0.8} ${H + 0.8}`} className="board-svg" key={stats.name + stats.layers.length}>
          <defs>
            <linearGradient id="woodg" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stopColor="#2a2622" />
              <stop offset="1" stopColor="#1c1a18" />
            </linearGradient>
            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="0.06" result="b" />
              <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          </defs>
          <rect x={0} y={0} width={W} height={H} rx={0.08} fill="url(#woodg)" stroke="#3a3f47" strokeWidth={0.02} />
          {svg.surround && (
            <path d={svg.surround} fill="#101318" fillRule="evenodd" className="layer-path"
                  style={{ animationDelay: "0ms", opacity: hover === null || hover === -1 ? 1 : 0.35 }}
                  onMouseEnter={() => setHover(-1)} onMouseLeave={() => setHover(null)} />
          )}
          {svg.layers.map((d, i) => (
            <path
              key={i}
              d={d}
              fill={stats.layers[i].color}
              fillRule="evenodd"
              className="layer-path"
              style={{
                animationDelay: `${i * 70}ms`,
                opacity: hover === null || hover === i ? 1 : 0.18,
              }}
              filter={hover === i ? "url(#glow)" : undefined}
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
            />
          ))}
          <path d={svg.outline} fill="none" stroke="#e9fffb" strokeWidth={0.012} opacity={0.9} pointerEvents="none" />
          {/* rulers */}
          <g className="ruler" fontSize={0.18}>
            <line x1={0} y1={-0.2} x2={W} y2={-0.2} />
            <text x={W / 2} y={-0.26} textAnchor="middle">{W}″</text>
            <line x1={-0.2} y1={0} x2={-0.2} y2={H} />
            <text x={-0.26} y={H / 2} textAnchor="middle" transform={`rotate(-90 ${-0.26} ${H / 2})`}>{H}″</text>
          </g>
        </svg>
      </div>

      <div className="layer-list glass">
        <div className="list-head">
          <span>Layer</span><span>Depth</span><span>Elevation</span><span>Area</span>
        </div>
        {stats.layers.map((l) => (
          <div
            key={l.index}
            className={"layer-row" + (hover === l.index ? " active" : "")}
            onMouseEnter={() => setHover(l.index)}
            onMouseLeave={() => setHover(null)}
          >
            <span className="swatch-cell">
              <i className="swatch" style={{ background: l.color }} />
              {l.index === 0 ? "Top" : `L${l.index}`}
            </span>
            <span className="mono">{l.index === 0 ? "uncut" : fmtIn(l.depth_in)}</span>
            <span className="mono dim">{Math.round(l.elev_m[0]).toLocaleString()}–{fmtM(l.elev_m[1])}</span>
            <span className="area">
              <b style={{ width: `${(l.area_in2 / maxArea) * 100}%`, background: l.color }} />
              <em className="mono">{l.area_in2.toFixed(1)} in²</em>
            </span>
          </div>
        ))}
        {stats.surround_depth_in != null && (
          <div className={"layer-row" + (hover === -1 ? " active" : "")}
               onMouseEnter={() => setHover(-1)} onMouseLeave={() => setHover(null)}>
            <span className="swatch-cell"><i className="swatch" style={{ background: "#101318" }} />Surround</span>
            <span className="mono">{fmtIn(stats.surround_depth_in)}</span>
            <span className="mono dim">outside region</span>
            <span />
          </div>
        )}
      </div>
    </div>
  );
}
