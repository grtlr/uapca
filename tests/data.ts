import * as studentGrades from '../data/student_grades.json';

import { describe, it } from 'mocha';
import { expect } from 'chai';
import { Matrix } from 'ml-matrix';
import { MultivariateNormal, UaPCA } from '../src/index.ts';

describe('Student dataset', () => {
    const dists = studentGrades.distributions.map(d => new MultivariateNormal(d.mean, d.cov));

    it('should load the dataset', () => {
        expect(dists.length).to.eql(6);
        expect(dists[0].mean()).to.eql(Matrix.rowVector([15, 12.285714285714285, 14.095238095238097, 15]));
    });

    it('should work with UaPCA', () => {
        const pca = UaPCA.fit(dists);
        const transformed = pca.transform(dists, 2);
        expect(transformed.length).to.eql(6);
        expect(transformed[0].mean().columns).to.eql(2);
    });
});
