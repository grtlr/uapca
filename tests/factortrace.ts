import {MultivariateNormal, getFactorTracer} from '../src/index.ts';
import {describe, it} from 'mocha';
import {StudentExample, TracePoint} from "./factortrace-data";

import {expect} from 'chai';
import {Matrix} from 'ml-matrix';

describe('FactorTracer', () => {
    it('factor tracer should return the correct projected unit vectors for a certain scaling factor', () => {
        const studentExample = new StudentExample();

        const dists = studentExample.means.map((m, i) => {
            const c = studentExample.covs[i];
            return new MultivariateNormal(m, c);
        });

        const tracer = getFactorTracer(dists, 2);

        studentExample.expected2dTracePoints.forEach((etp: TracePoint) => {
            const projectedUnitVectors: Array<Matrix> = tracer(etp.scale);

            projectedUnitVectors.forEach((puv: Matrix, i: number) => {
                const expectedPuv: Matrix = etp.projectedUnitVectors[i];

                const puvArray = puv.to1DArray();
                const expectedPuvArray = expectedPuv.to1DArray();

                puvArray.forEach((value: number, i: number) => {
                    const expectedValue: number = expectedPuvArray[i];

                    expect(Math.abs(value)).to.be.closeTo(Math.abs(expectedValue), 0.000000000001);
                });
            });
        });
    });
});
