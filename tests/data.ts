import { MultivariateNormal, UaPCA } from '../src/index.ts';
import { describe, it } from 'mocha';

import { expect } from 'chai';
import { Matrix } from 'ml-matrix';

import * as student_grades from '../data/student_grades.json';

describe('Student dataset', () => {
    it('should find the correct projection', () => {

       let dists = student_grades.distributions.map(d => new MultivariateNormal(
           new Matrix(d.meanVec),
           new Matrix(d.covMat)
       ));

       expect(dists.length).to.eql(6);
       expect(dists[0].mean()).to.eql(Matrix.columnVector([15, 12.29, 14.1, 15]));
    });
});
