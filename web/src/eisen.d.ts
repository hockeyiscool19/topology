/**
 * The eisensoftware platform's shared usage script (`/telemetry/v1.js`, loaded once in index.html) defines
 * `window.eisen` on eisensoftware.com only, and sends nothing until the person opts in on `/privacy`.
 * Map and 3D-canvas moments call it with a literal name and optional chaining: `window.eisen?.track("draw.done")`.
 * Contract: the usage-analytics skill (hockeyiscool19/monorepo, plugins/eisen-platform/skills/usage-analytics).
 */
interface Window {
  eisen?: { track(name: string): void };
}
