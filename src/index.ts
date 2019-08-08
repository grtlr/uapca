import { Matrix } from 'ml-matrix';

interface Distribution {
    mean(): Matrix;
    covariance(): Matrix;
}

export class MultivariateNormal implements Distribution {
    private meanVec: Matrix;
    private covMat: Matrix;
    public constructor(mean: Matrix, covMat: Matrix) {
        this.meanVec = mean;
        this.covMat = covMat;
    }

    public mean(): Matrix {
        return this.meanVec;
    }

    public covariance(): Matrix {
        return this.covMat;
    }
}

export class StandardNormal extends MultivariateNormal {
    public constructor(nDims: number) {
        super(Matrix.zeros(nDims, 1), Matrix.eye(nDims, nDims));
    }
}

export class PrincipalComponents {
    private lengths: Array<number>;
    private vectors: Matrix;
    public constructor(lengths: Array<number>, vectors: Matrix) {
        this.lengths = lengths;
        this.vectors = vectors;
    }
}

export class UaPCA {
    private nComponents: number;
    public constructor(nComponents: number) {
        this.nComponents = nComponents;
    }

    public fit(_distributions: Array<Distribution>): PrincipalComponents {
        return new PrincipalComponents([], new Matrix([[1]]));
    }
}
