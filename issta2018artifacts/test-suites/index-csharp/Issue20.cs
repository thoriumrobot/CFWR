using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class Issue20 {
    // An issue with LUB that results in losing information when unifying.
    protected int[] a, b;

    void test(int i, int j, bool flag) {
        Contract.Requires(i<a.Length);
        Contract.Requires(j<=a.Length&& j<=b.Length);
        int k = flag ? i : j;
        if(TestHelper.nondet()) Contract.Assert(k <= a.Length);
    }
}
