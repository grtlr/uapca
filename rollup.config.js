import commonjs from '@rollup/plugin-commonjs';
import resolve from 'rollup-plugin-node-resolve';
import eslint from '@rollup/plugin-eslint';
import { terser } from 'rollup-plugin-terser';
import typescript from '@rollup/plugin-typescript'

import pkg from './package.json'

const copyright = `// ${pkg.name} v${pkg.version} Copyright ${(new Date).getFullYear()} ${pkg.author.name}`;

export default [
    {
        input: 'src/index.ts',
        output: [
            {
                banner: copyright,
                name: pkg.name,
                file: 'dist/uapca.js',
                format: 'umd',
            },
        ],
        plugins: [
            eslint({
                throwOnWarning: true
            }),
            resolve({
                preferBuiltins: true,
                browser: true

            }),
            commonjs(),
            typescript(),
        ],
    },
    {
        input: 'src/index.ts',
        output: [
            {
                name: pkg.name,
                file: 'dist/uapca.min.js',
                format: 'umd',
            },
        ],
        plugins: [
            eslint({
                throwOnWarning: true
            }),
            resolve({
                preferBuiltins: true,
                browser: true

            }),
            typescript(),
            commonjs(),
            terser({output: {preamble: copyright}})
        ],
    }
]
