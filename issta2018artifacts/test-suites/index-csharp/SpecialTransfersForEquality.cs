using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;


class SpecialTransfersForEquality {

    void gteN1Test(int y) {
        Contract.Requires(y >= -1);
        int[] arr = new int[10];
        if (-1 != y) {
            int z = y;
            if(TestHelper.nondet()) Contract.Assert(z >= 0);
            if (z < 10) {
                int k = arr[z];
            }
        }
    }

    void nnTest(int i) {
        Contract.Requires(i >= 0);
        if (i != 0) {
            int m = i;
            if(TestHelper.nondet()) Contract.Assert(m > 0);
        }
    }
}
