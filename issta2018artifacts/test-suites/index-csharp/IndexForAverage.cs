// test case for issue 86: https://github.com/kelloggm/checker-framework/issues/86

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class IndexForAverage {

    public static void bug(int[] a, int i, int j) {
        Contract.Requires(i >= 0 && i < a.Length);
        Contract.Requires(j >= 0 && j < a.Length);
        int k = (i + j) / 2;
        if(TestHelper.nondet()) Contract.Assert(k >= 0 && k < a.Length);
    }

    public static void bug2(int[] a, int i, int j) {
        Contract.Requires(i >= 0 && i < a.Length);
        Contract.Requires(j >= 0 && j < a.Length);
        int k = ((i - 1) + j) / 2;
        if(TestHelper.nondet()) Contract.Assert(k < a.Length);
        // :: error: (assignment.type.incompatible)
        int h = ((i + 1) + j) / 2;
        if(TestHelper.nondet()) Contract.Assert(h < a.Length);
    }
}
