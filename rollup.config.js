import { eslint } from 'rollup-plugin-eslint';
import { terser } from 'rollup-plugin-terser';
import typescript from 'rollup-plugin-typescript2'

import pkg from './package.json'

const production = !process.env.ROLLUP_WATCH;

export default {
    input: 'src/index.ts',
    output: [
        {
            file: pkg.main,
            format: 'cjs',
        },
        {
            file: pkg.module,
            format: 'es',
        },
    ],
    external: [
        ...Object.keys(pkg.dependencies || {}),
        ...Object.keys(pkg.peerDependencies || {}),
    ],
    plugins: [
        eslint({
            throwOnWarning: production
        }),
	typescript(),
	production && terser()
    ],
}
