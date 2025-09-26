using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class Errors {

    void test() {
        int[] arr = new int[5];

        // unsafe
        int n1p = -1;
        if(TestHelper.nondet()) Contract.Assert(n1p >= -1);
        int u = -10;
        if(TestHelper.nondet()) Contract.Assert(true);

        // safe
        int nn = 0;
        if(TestHelper.nondet()) Contract.Assert(nn >= 0);
        int p = 1;
        if(TestHelper.nondet()) Contract.Assert(p > 0);

        // :: error: (array.access.unsafe.low)
        int a = arr[n1p];

        // :: error: (array.access.unsafe.low)
        int b = arr[u];

        int c = arr[nn];
        int d = arr[p];
    }
}
// a comment
