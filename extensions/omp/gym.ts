/**
 * /gym — overnight skill gym for OMP (omp-gym binary)
 *
 * Install:
 *   cp extensions/omp/gym.ts ~/.omp/agent/extensions/
 *   cargo install --path crates/omp-gym
 *
 * Usage in OMP:
 *   /gym status
 *   /gym dry-run
 *   /gym run
 *   /gym overnight
 *   /gym overnight off
 *   /gym adopt
 *   /gym doctor
 */
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";

function resolveGymBin(): string {
  const env = process.env.OMP_GYM_BIN;
  if (env && existsSync(env)) return env;

  const home = homedir();
  const candidates = [
    join(home, ".cargo", "bin", "omp-gym"),
    join(home, "Desktop", "omp-gym", "target", "release", "omp-gym"),
    join(home, "Desktop", "omp-gym", "target", "debug", "omp-gym"),
    "omp-gym",
  ];
  for (const c of candidates) {
    if (c === "omp-gym") return c;
    if (existsSync(c)) return c;
  }
  return "omp-gym";
}

function runGym(args: string[], cwd: string): { code: number; out: string; err: string } {
  const bin = resolveGymBin();
  const res = spawnSync(bin, args, {
    cwd,
    encoding: "utf8",
    env: process.env,
  });
  return {
    code: res.status ?? 1,
    out: (res.stdout ?? "").trim(),
    err: (res.stderr ?? "").trim() || (res.error ? String(res.error) : ""),
  };
}

function helpText(): string {
  return [
    "omp-gym — session harvester/miner prototype for OMP",
    "",
    "Commands:",
    "  /gym status              Show harvest/run/proposal state",
    "  /gym dry-run             Harvest this project's OMP sessions and mine tasks",
    "  /gym run                 Stage mock proposal metadata; does not edit a skill",
    "  /gym overnight           Schedule daily mock snapshots at 02:15 (macOS)",
    "  /gym overnight off       Remove the launchd schedule",
    "  /gym adopt               Reserved; v0.1 refuses all generated mock proposals",
    "  /gym doctor              Show project, session, and binary paths",
    "  /gym help                This text",
    "",
    "No replay, optimization, validation, or skill editing exists in v0.1.",
    "Set OMP_GYM_BIN to override the binary path.",
  ].join("\n");
}

export default function gymExtension(pi: ExtensionAPI) {
  pi.registerCommand("gym", {
    description: "Harvest project sessions and stage mock gym artifacts",
    handler: async (args, ctx) => {
      const cwd = ctx.cwd;
      const raw = (args ?? "").trim();
      if (!raw || raw === "help" || raw === "-h" || raw === "--help") {
        ctx.ui.notify(helpText(), "info");
        return;
      }

      // Support: /gym overnight off | /gym run -- --target-skill ...
      const parts = raw.split(/\s+/);
      let sub = parts[0] ?? "help";
      let rest = parts.slice(1);

      // allow `/gym overnight off`
      if (sub === "overnight") {
        if (rest[0] === "off") {
          const r = runGym(["schedule", "--off", "--project", cwd], cwd);
          ctx.ui.notify(r.out || r.err || `exit ${r.code}`, r.code === 0 ? "info" : "error");
          return;
        }
        // default overnight schedule
        const r = runGym(["schedule", "--hour", "2", "--minute", "15", "--project", cwd, ...rest], cwd);
        ctx.ui.notify(r.out || r.err || `exit ${r.code}`, r.code === 0 ? "info" : "error");
        return;
      }

      // strip leading -- used as separator
      if (rest[0] === "--") rest = rest.slice(1);

      const map: Record<string, string[]> = {
        status: ["status"],
        "dry-run": ["dry-run"],
        dryrun: ["dry-run"],
        run: ["run"],
        adopt: ["adopt"],
        doctor: ["doctor"],
        schedule: ["schedule"],
      };

      const base = map[sub];
      if (!base) {
        ctx.ui.notify(`Unknown /gym subcommand: ${sub}\n\n${helpText()}`, "error");
        return;
      }

      const argv = [...base, "--project", cwd, ...rest];
      const r = runGym(argv, cwd);
      const body = [r.out, r.err].filter(Boolean).join("\n");
      ctx.ui.notify(body || `omp-gym exited ${r.code}`, r.code === 0 ? "info" : "error");
    },
  });
}
