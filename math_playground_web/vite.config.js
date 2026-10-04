/** @type {import('vite').UserConfig} */
export default {
  // The wasm package is built with `wasm-pack build --target web` and loaded via an explicit URL,
  // so no wasm/top-level-await plugins are needed.
  server: { fs: { allow: [".."] } },
  build: { target: "es2020" },
};
