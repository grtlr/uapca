import * as studentGrades from '../data/student_grades.json';

import { describe, it } from 'mocha';
import { expect } from 'chai';
import { Matrix } from 'ml-matrix';
import { MultivariateNormal } from '../src/index.ts';
import {UaPCA} from "../src/index";

describe('Student dataset', () => {
    const dists = studentGrades.distributions.map(d => new MultivariateNormal(
        d.mean,
        new Matrix(d.cov)
    ));

    it('should load the dataset', () => {
        expect(dists.length).to.eql(6);
        expect(dists[0].mean()).to.eql(new Matrix([[15, 12.285714285714285, 14.095238095238097, 15]]));
    });

    // TODO: Add test for UaPCA.fit()
});
