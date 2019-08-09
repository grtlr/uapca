import { describe, it } from 'mocha';
import { MultivariateNormal, Point, UaPCA } from '../src/index.ts';

import { expect } from 'chai';
import { Matrix } from 'ml-matrix';

describe('Distribution', () => {
    describe('StandardNormal', () => {
        const sn = MultivariateNormal.standard(2);

        it('should return zero as mean', () => {
            expect(sn.mean()).to.eql(new Matrix([[0], [0]]));
        });

        it('should return eye as covariance', () => {
            expect(sn.covariance()).to.eql(new Matrix([[1, 0], [0, 1]]));
        });
    });
});

describe('Projection', () => {
    describe('Points', () => {
        it('should be projected correctly', () => {
            const v1 = new Point([1, 1, 1]);
            const v2 = new Point([1, 42, 42]);
            const projmat = new Matrix([[1, 0, 0], [0, 1, 0]]);
            const pv1 = v1.project(projmat);
            const pv2 = v2.project(projmat);
            expect(pv1).to.eql(new Point([1, 1]));
            expect(pv2).to.eql(new Point([1, 42]));
        });
    });

    describe('StandardNormal', () => {
        it('should be projected correctly', () => {
            const sn3 = MultivariateNormal.standard(3);
            const projmat = new Matrix([[1, 0, 0], [0, 1, 0]]);
            const psn3 = sn3.project(projmat);
            expect(psn3).to.eql(MultivariateNormal.standard(2));
        });
    });

    describe('correlated MultivariateNormal', () => {
        it('should be projected correctly', () => {
            const mean = Matrix.zeros(3, 1);
            const covMat = new Matrix([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]);
            const mvn = new MultivariateNormal(mean, covMat);
            const projmat = new Matrix([[1, 0, 0], [0, 1, 0]]);
            const result = mvn.project(projmat);
            expect(result).to.eql(new MultivariateNormal(
                Matrix.zeros(2, 1),
                new Matrix([[1, 0.5], [0.5, 1]])
            ));
        });
    });

    describe('non-zero mean MultivariateNormal', () => {
        it('should be projected correctly', () => {
            const mean = Matrix.columnVector([1, 42, 0]);
            const covMat = new Matrix([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]);
            const mvn = new MultivariateNormal(mean, covMat);
            const projmat = new Matrix([[1, 0, 0], [0, 1, 0]]);
            const result = mvn.project(projmat);
            expect(result).to.eql(new MultivariateNormal(
                Matrix.columnVector([1, 42]),
                new Matrix([[1, 0.5], [0.5, 1]])
            ));
        });
    });
});

describe('UaPCA fitting', () => {
    const means = [
        [2, 0, 0],
        [-2, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
    ];
    describe('an isometric MultivariateNormal', () => {
        it('should return the same principal components as regular PCA', () => {
            const iso = Matrix.eye(3, 3);
            const dists = means.map(m => new MultivariateNormal(Matrix.columnVector(m), iso));
            const points = means.map(m => new Point(m));

            const pcs1 = UaPCA.fit(dists);
            const pcs2 = UaPCA.fit(points);

            expect(pcs1.lengths.length).to.eql(3);
            expect(pcs2.lengths.length).to.eql(3);
            expect(pcs1.vectors).to.eql(pcs2.vectors);
        });
    });
});
