using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

/*@SuppressWarnings("lowerbound")*/
class CombineFacts {
    void test(int[] a1) {
        int len = a1.Length - 1;
        if(TestHelper.nondet()) Contract.Assert(len < a1.Length);
        int[] a2 = new int[len];
        a2[len - 1] = 1;
        a1[len] = 1;

        // This access should issue an error.
        // :: error: (array.access.unsafe.high)
        a2[len] = 1;
    }
}
