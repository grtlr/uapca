import { arithmeticMean, MultivariateNormal, Sampler, UaPCA } from '../src/index.ts';
import { describe, it } from 'mocha';
import { expect } from 'chai';
import { Matrix } from 'ml-matrix';
import { Point } from './util';

describe('StandardNormal', () => {
    const sn = MultivariateNormal.standard(2);

    it('should return zero as mean', () => {
        expect(sn.mean()).to.eql(Matrix.rowVector([0, 0]));
    });

    it('should return eye as covariance', () => {
        expect(sn.covariance()).to.eql(new Matrix([[1, 0], [0, 1]]));
    });
});

describe('MultivariateNormal', () => {
    it('should allow both constructor types', () => {
        const a = new MultivariateNormal([0, 0], [[1, 0], [0, 1]]);
        const b = new MultivariateNormal(Matrix.rowVector([0, 0]), Matrix.diagonal([1, 1]));
        expect(a).to.eql(b);
    });
});

describe('Projection', () => {
    it('StandardNormal should be projected correctly', () => {
        const sn3 = MultivariateNormal.standard(3);
        const projmat = new Matrix([[1, 0, 0], [0, 1, 0]]);
        const psn3 = sn3.project(projmat.transpose());
        expect(psn3).to.eql(MultivariateNormal.standard(2));
    });

    it('correlated MultivariateNormal should be projected correctly', () => {
        const mean = Matrix.zeros(1, 3);
        const covMat = new Matrix([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]);
        const mvn = new MultivariateNormal(mean, covMat);
        const projmat = new Matrix([[1, 0, 0], [0, 1, 0]]);
        const result = mvn.project(projmat.transpose());
        expect(result).to.eql(new MultivariateNormal(
            Matrix.zeros(1, 2),
            new Matrix([[1, 0.5], [0.5, 1]])
        ));
    });

    it('non-zero mean MultivariateNormal should be projected correctly', () => {
        const mvn = new MultivariateNormal(
            [1, 42, 0],
            [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]
        );
        const projmat = new Matrix([[1, 0, 0], [0, 1, 0]]);
        const result = mvn.project(projmat.transpose());
        expect(result).to.eql(new MultivariateNormal([1, 42], [[1, 0.5], [0.5, 1]]));
    });
});

describe('UaPCA', () => {
    it('should return the same principal components as regular PCA', () => {
        const means = [
            [2, 0, 0],
            [-2, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
        ];
        const iso = Matrix.eye(3, 3);
        const dists = means.map(m => new MultivariateNormal(m, iso));

        const zero = Matrix.zeros(3, 3);
        const dists0 = means.map(m => new MultivariateNormal(m, zero));

        const pca1 = UaPCA.fit(dists);
        const pca2 = UaPCA.fit(dists0);

        expect(pca1.lengths.length).to.eql(3);
        expect(pca1.vectors).to.eql(pca2.vectors);
    });

    it('should yield same result as sampling', () => {
        const mean = [42, 0];
        const cov = new Matrix([[1, 0.5], [0.5, 1]]);
        const gauss = new MultivariateNormal(mean, cov);
        const means = (new Sampler(gauss)).sampleN(3);

        const dists = means.map(m => new MultivariateNormal(m, cov));
        const pca1 = UaPCA.fit(dists);

        const zero = Matrix.zeros(2, 2);
        const points = dists.map(d => (new Sampler(d)).sampleN(25000)).flat()
            .map(m => new MultivariateNormal(m, zero));
        const pca2 = UaPCA.fit(points);

        expect(pca1.vectors.rows).to.be.eql(2);
        expect(pca1.vectors.columns).to.be.eql(2);

        expect(pca1.lengths[0]).to.be.closeTo(pca2.lengths[0], 0.1);
        expect(pca1.lengths[1]).to.be.closeTo(pca2.lengths[1], 0.1);

        expect(pca1.vectors.get(0, 0)).to.be.closeTo(pca2.vectors.get(0, 0), 0.1);
        expect(pca1.vectors.get(1, 0)).to.be.closeTo(pca2.vectors.get(1, 0), 0.1);
        expect(pca1.vectors.get(0, 1)).to.be.closeTo(pca2.vectors.get(0, 1), 0.1);
        expect(pca1.vectors.get(1, 1)).to.be.closeTo(pca2.vectors.get(1, 1), 0.1);
    });

    it('should yield same result as regular PCA when scaling with 0', () => {
        const mean = Matrix.rowVector([42, 0]);
        const cov = new Matrix([[1, 0.5], [0.5, 1]]);
        const gauss = new MultivariateNormal(mean, cov);
        const means = (new Sampler(gauss)).sampleN(3);

        const dists = means.map(m => new MultivariateNormal(m, cov));
        const pca1 = UaPCA.fit(dists, 0);

        const zero = Matrix.zeros(2, 2);
        const points = means.map(m => new MultivariateNormal(m, zero));
        const pca2 = UaPCA.fit(points);

        expect(pca1.vectors.rows).to.be.eql(2);
        expect(pca1.vectors.columns).to.be.eql(2);

        expect(pca1.lengths[0]).to.be.closeTo(pca2.lengths[0], 0.1);
        expect(pca1.lengths[1]).to.be.closeTo(pca2.lengths[1], 0.1);

        expect(pca1.vectors.get(0, 0)).to.be.closeTo(pca2.vectors.get(0, 0), 0.1);
        expect(pca1.vectors.get(1, 0)).to.be.closeTo(pca2.vectors.get(1, 0), 0.1);
        expect(pca1.vectors.get(0, 1)).to.be.closeTo(pca2.vectors.get(0, 1), 0.1);
        expect(pca1.vectors.get(1, 1)).to.be.closeTo(pca2.vectors.get(1, 1), 0.1);
    });

    it('fitTransform should yield the correct dimensions', () => {
        const means = [
            [2, 0, 0],
            [-2, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
        ];
        const cov = new Matrix([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]);
        const dists = means.map(m => new MultivariateNormal(m, cov));
        const transformed = UaPCA.fit(dists).transform(dists, 2);

        expect(transformed.length).to.be.eql(4);
        expect(transformed[0].mean().columns).to.be.eql(2);
    });
});

describe('Sampler', () => {
    it('should create the right amount of samples', () => {
        const dist = MultivariateNormal.standard(3);
        const samples = (new Sampler(dist)).sampleN(1000);
        expect(samples.length).to.eql(1000);
    });

    it('should have correct dimensions', () => {
        const dist = MultivariateNormal.standard(3);
        const samples = (new Sampler(dist)).sampleN(1000);
        expect(samples[0].length).to.eql(3);
    });

    it('samples should have the same arithmetic mean', () => {
        const mean = Matrix.columnVector([42, 0]);
        const cov = new Matrix([[1, 0.5], [0.5, 1]]);
        const dist = new MultivariateNormal(mean, cov);
        const samples = (new Sampler(dist)).sampleN(100000)
            .map(s => Matrix.columnVector(s));
        const sampleMean = arithmeticMean(samples);
        expect(sampleMean.get(0, 0)).to.be.closeTo(mean.get(0, 0), 0.01);
        expect(sampleMean.get(1, 0)).to.be.closeTo(mean.get(1, 0), 0.01);
    });

    it('samples should have the same sample covariance matrix', () => {
        const count = 100000;
        const mean = [42, 0];
        const cov = new Matrix([[1, 0.5], [0.5, 1]]);
        const dist = new MultivariateNormal(mean, cov);
        const samples = (new Sampler(dist)).sampleN(count);
        const sampleMean = arithmeticMean(samples.map(s => Matrix.rowVector(s)));

        const sampleMat = new Matrix(samples);
        const sampleCov = sampleMat.transpose().mmul(sampleMat);
        sampleCov.div(count);
        sampleCov.sub(sampleMean.transpose().mmul(sampleMean));
        expect(sampleCov.rows).to.be.eql(2);
        expect(sampleCov.columns).to.be.eql(2);
        expect(sampleCov.get(0, 0)).to.be.closeTo(cov.get(0, 0), 0.01);
        expect(sampleCov.get(1, 0)).to.be.closeTo(cov.get(1, 0), 0.01);
        expect(sampleCov.get(0, 1)).to.be.closeTo(cov.get(0, 1), 0.01);
        expect(sampleCov.get(1, 1)).to.be.closeTo(cov.get(1, 1), 0.01);
    });
});
