import * as studentGrades from '../data/student_grades.json';

import { describe, it } from 'mocha';
import { Matrix, MultivariateNormal, UaPCA } from '../src/index.ts';
import { expect } from 'chai';

describe('Student dataset', () => {
    const dists = studentGrades.distributions.map(d => new MultivariateNormal(d.mean, d.cov));

    it('should load the dataset', () => {
        expect(dists.length).to.eql(6);
        const projected = Matrix.rowVector([15, 12.285714285714285, 14.095238095238097, 15]);
        expect(dists[0].mean()).to.eql(projected);
    });

    it('should work with UaPCA', () => {
        const pca = UaPCA.fit(dists);
        const transformed = pca.transform(dists, 2);
        expect(transformed.length).to.eql(6);
        expect(transformed[0].mean().columns).to.eql(2);
    });
});
