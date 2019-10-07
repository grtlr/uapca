import { Distribution, Projection } from '../src/index.ts';
import { Matrix } from 'ml-matrix';

export class Point implements Distribution, Projection {
    private data: Matrix;
    public constructor(data: Array<number>) {
        this.data = Matrix.rowVector(data);
    }

    public mean(): Matrix {
        return this.data;
    }

    public covariance(): Matrix {
        return Matrix.zeros(this.data.columns, this.data.columns);
    }

    public project(projectionMatrix: Matrix): Point {
        return new Point(this.data.mmul(projectionMatrix).getRow(0));
    }
}

