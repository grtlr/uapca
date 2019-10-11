import * as d3 from 'd3-array';
import { UaPCA, Matrix } from './uapca';

export class Trace {
    private samples: uapca.Matrix;
    public constructor(samples: uapca.Matrix) {
        this.samples = samples;
    }

    public points(): Array<Array<number>> {
        return this.samples.to2DArray();
    }

    public pointsFlipped(): Array<Array<number>> {
        return this.samples.mul(-1).to2DArray();
    }
}

// TODO: Consider providing all options to factor traces as a `FactorTracesOption` interface
export class FactorTraces {
    private min: number;
    private max: number;
    private N: number;
    private components: number;
    private traceSamples: Array<uapca.Matrix>;

    public constructor(
        distributions: Array<uapca.Distribution>,
        numSamples?: number,
        min?: number,
        max?: number,
        components?: number
    ) {
        this.N = numSamples ? numSamples : 100;
        this.min = min ? min : 0;
        this.max = max ? max : 1000;
        this.components = components ? components : 2;

        // We sample the scaling of uncertainty for the factor traces quadratically
        const yMin = this.min ** 2;
        const yMax = this.max ** 2;
        const step = (yMax - yMin) / this.N;
        const sampleAt = d3.range(this.N).map(i => Math.sqrt(yMin + i * step));

        // We need to fit the aligned version of each PCA because components might flip otherwise.
        const PCAs = sampleAt.map(t => uapca.UaPCA.fit(distributions, t).aligned());

        const nDims = distributions[0].mean().columns;
        this.traceSamples = PCAs.map(pca => {
            const Pt = pca.projectionMatrix(this.components).transpose();
            return uapca.Matrix.eye(nDims, nDims).mmul(Pt);
        });
    }

    public getTrace(dimension: number): Trace {
        // initialize
        const projected = this.traceSamples[0];
        const trace = uapca.Matrix.zeros(this.N, this.components);
        trace.setRow(0, projected.getRowVector(dimension));

        for (let i = 1; i < this.traceSamples.length; ++i) {
            const projected = this.traceSamples[i];

            const a = projected.getRowVector(dimension);
            const b = projected.getRowVector(dimension).mul(-1);
            const distA = uapca.Matrix.sub(a, trace.getRowVector(i - 1));
            const distB = uapca.Matrix.sub(b, trace.getRowVector(i - 1));

            if (distA.dot(distA) < distB.dot(distB)) {
                trace.setRow(i, a);
            } else {
                trace.setRow(i, b);
            }
        }
        return new Trace(trace);
    }
}
