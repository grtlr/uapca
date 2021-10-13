import * as studentGrades from '../data/student_grades.json';

import { describe, it } from 'mocha';
import { FactorTraces, MultivariateNormal } from '../src/index';
import { expect } from 'chai';

describe('FactorTraces', () => {
    const dists = studentGrades.distributions.map(d => new MultivariateNormal(d.mean, d.cov));

    it('should have the correct number of samples', () => {
        const numSamples = 100;
        const traces = new FactorTraces(dists, numSamples);
        const a = traces.getTrace(0);
        expect(a.points().length).to.be.eql(numSamples);
    });

    it('should have the correct number of components', () => {
        const numSamples = 2;
        const traces = new FactorTraces(dists, numSamples);
        const a = traces.getTrace(0);
        expect(a.points()[0].length).to.be.eql(2);
    });

    it('should have the correct number of flipped points', () => {
        const numSamples = 100;
        const traces = new FactorTraces(dists, numSamples);
        const a = traces.getTrace(0);
        expect(a.pointsFlipped().length).to.be.eql(a.points().length);
    });
});

