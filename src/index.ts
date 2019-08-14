import * as d3 from 'd3-random';
import { EigenvalueDecomposition, IRandomOptions, Matrix } from 'ml-matrix';

export interface Distribution {
    mean(): Matrix;
    covariance(): Matrix;
}

export interface Projection {
    /**
     * Projects a distribution onto a lower dimensional subspace
     * defined by the dimensions of the `projectionMatrix`.
     * @param {Matrix} projectionMatrix - Defined as row matrix.
     */
    project(projectionMatrix: Matrix): Distribution;
}

export class MultivariateNormal implements Distribution, Projection {
    private meanVec: Matrix;
    private covMat: Matrix;
    public constructor(meanVec: Array<number> | Matrix, covMat: Array<Array<number>> | Matrix) {
        this.meanVec = (meanVec instanceof Matrix) ? meanVec : Matrix.columnVector(meanVec);
        this.covMat = (covMat instanceof Matrix) ? covMat : new Matrix(covMat);
    }

    public static standard(nDims: number): MultivariateNormal {
        return new MultivariateNormal(Matrix.zeros(nDims, 1), Matrix.eye(nDims, nDims));
    }

    public mean(): Matrix {
        return this.meanVec;
    }

    public covariance(): Matrix {
        return this.covMat;
    }

    public project(projectionMatrix: Matrix): MultivariateNormal {
        const newMean = projectionMatrix.mmul(this.meanVec);
        const newCovMat = projectionMatrix.mmul(this.covMat).mmul(projectionMatrix.transpose());
        return new MultivariateNormal(newMean, newCovMat);
    }
}

class RandomStdNormal implements IRandomOptions {
    public random: () => number;
    public constructor() {
        this.random = d3.randomNormal();
    }
}

export class Sampler {
    private mean: Matrix;
    private A: Matrix;
    private gen: RandomStdNormal;
    public constructor(distribution: MultivariateNormal) {
        this.mean = distribution.mean().transpose();
        const eigen = new EigenvalueDecomposition(distribution.covariance());
        const q = eigen.eigenvectorMatrix.transpose();
        const lambda = Matrix.diag(eigen.realEigenvalues.map(x => Math.sqrt(x)));
        this.A = lambda.mmul(q);
        this.gen = new RandomStdNormal();
    }

    public sampleN(count: number): Array<Array<number>> {
        // Z is transposed because A is also transposed
        const Z = Matrix.random(count, this.dims(), this.gen);
        const res = Z.mmul(this.A).addRowVector(this.mean);
        return res.to2DArray();
    }

    private dims(): number {
        return this.mean.columns;
    }
}

export class Point implements Distribution, Projection {
    private data: Matrix;
    public constructor(data: Array<number>) {
        this.data = Matrix.columnVector(data);
    }

    public mean(): Matrix {
        return this.data;
    }

    public covariance(): Matrix {
        return Matrix.zeros(this.data.rows, this.data.rows);
    }

    public project(projectionMatrix: Matrix): Point {
        return new Point(projectionMatrix.mmul(this.data).getColumn(0));
    }
}

function outerProduct(x: Matrix): Matrix {
    return x.mmul(x.transpose());
}

export function arithmeticMean(matrices: Array<Matrix>): Matrix {
    const nrows = matrices[0].rows;
    const ncols = matrices[0].columns;
    const N = matrices.length;
    const sum = matrices.reduce((acc, m) => acc.add(m), Matrix.zeros(nrows, ncols));
    return sum.div(N);
}

function centering(distributions: Array<Distribution>): Matrix {
    const v = arithmeticMean(distributions.map(d => d.mean()));
    return outerProduct(v);
}

export class UaPCA {
    public readonly lengths: Array<number>;
    public readonly vectors: Matrix; // row matrix!

    private constructor(lengths: Array<number>, vectors: Matrix) {
        this.lengths = lengths;
        this.vectors = vectors;
    }

    public static fit(
        distributions: Array<Distribution>,
        scale: number = 1.0,
    ): UaPCA {
        const center: Matrix = centering(distributions);
        const empericalCov: Matrix = arithmeticMean(distributions.map(d => {
            return outerProduct(d.mean()).add(Matrix.mul(d.covariance(), scale * scale))
                .sub(center);
        }));

        // Compute components and sort by eigenvalues
        const e = new EigenvalueDecomposition(empericalCov);
        const evals = e.realEigenvalues;
        const evecs = e.eigenvectorMatrix.transpose().to2DArray();

        const pairs: Array<[number, Array<number>]> = evals.map((e, i) => [e, evecs[i]]);
        const comps = pairs.sort((a, b) => b[0] - a[0]);

        return new UaPCA(comps.map(d => d[0]), new Matrix(comps.map(v => v[1])));
    }

    public transform(
        distributions: Array<Distribution & Projection>,
        components: number,
    ): Array<Distribution> {
        const projMat = new Matrix(this.vectors.to2DArray().slice(0, components));
        return distributions.map(d => d.project(projMat));
    }
}

export { Matrix };
