import * as studentGrades from '../data/student_grades.json';
import * as studentGradesExpected from '../tests/data/student_grades_expected.json';

import { describe, it } from 'mocha';
import {
    getFactorTracer,
    getTraceIterator,
    MultivariateNormal,
    Point,
    ProjectedUnitVectors,
    TracePoint,
} from '../src/index';

import { expect } from 'chai';
import { Matrix } from 'ml-matrix';

describe('FactorTracer', () => {
    it('factor tracer should return the correct projected unit vectors for a certain scaling factor', () => {

        /*function flatten(arr) {
            return arr.reduce(function (flat, toFlatten) {
                return flat.concat(Array.isArray(toFlatten) ? flatten(toFlatten) : toFlatten);
            }, []);
        }

        studentGradesExpected.expected2dTracePoints.forEach(d => {
           d.data = flatten(d.mean);
        });

        console.log(JSON.stringify(studentGrades));*/

        const dists = studentGrades.distributions.map(d => new MultivariateNormal(
            d.mean,
            new Matrix(d.cov)
        ));

        const etps = studentGradesExpected.expected2dTracePoints.map(d => {
            const puvs = d.projectedUnitVectors.map(puv => new Point(puv.data));
            return {
                scale: d.projectedUnitVectors.scale,
                projectedUnitVectors: puvs,
            };
        });

        const tracer = getFactorTracer(dists, 2);

        etps.forEach((etp: TracePoint) => {
            const projectedUnitVectors: Array<Point> = tracer(etp.scale);

            projectedUnitVectors.forEach((puv: Point, i: number) => {
                const expectedPuv: Point = etp.projectedUnitVectors[i];

                const puvArray = puv.getData();
                const expectedPuvArray = expectedPuv.getData();

                puvArray.forEach((value: number, i: number) => {
                    const expectedValue: number = expectedPuvArray[i];

                    expect(Math.abs(value)).to.be.closeTo(Math.abs(expectedValue), 0.000000000001);
                });
            });
        });
    });
});

describe('FactorTraceGenerator', () => {
    it('in progress', () => {
        const dists = studentGrades.distributions.map(d => new MultivariateNormal(
            d.mean,
            new Matrix(d.cov)
        ));

        const tracer = getFactorTracer(dists, 2);
        const traceIterator = getTraceIterator(tracer, 1000);

        const trace = [...traceIterator].map((tp: TracePoint) => {
            const t = tp.scale;

            const puvs = tp.projectedUnitVectors.reduce((acc: Array<Array<number>>, curr: Point) => {
                return [...acc, curr.getData()];
            }, []);

            return { t, puvs };
        });

        // TODO: Compare to expected values
    });
});
