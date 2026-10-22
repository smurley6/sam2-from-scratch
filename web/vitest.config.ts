import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
    // The parity suites each load the graphs; running them one at a time keeps memory and
    // the per-frame timings sane.
    fileParallelism: false,
    testTimeout: 15 * 60 * 1000,
    hookTimeout: 5 * 60 * 1000,
  },
});
