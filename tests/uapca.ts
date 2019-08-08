import { describe, it } from 'mocha';
import { expect } from 'chai';

import { Matrix } from 'ml-matrix';
import { StandardNormal } from '../src/index.ts';

describe('StandardNormal', () => {
    describe('mean', () => {
        it('should return zero mean', () => {
            const sn = new StandardNormal(2);
            expect(sn.mean()).to.eql(new Matrix([[0], [0]]));
        });
    });
});
