import * as studentGrades from '../data/student_grades.json';

import { describe, it } from 'mocha';
import { expect } from 'chai';
import { Matrix } from 'ml-matrix';
import { MultivariateNormal } from '../src/index.ts';

describe('Student dataset', () => {
    const dists = studentGrades.distributions.map(d => new MultivariateNormal(
        new Matrix(d.mean),
        new Matrix(d.cov)
    ));

    it('should load the dataset', () => {
        expect(dists.length).to.eql(6);
        expect(dists[0].mean()).to.eql(Matrix.columnVector([15, 12.29, 14.1, 15]));
    });
});
