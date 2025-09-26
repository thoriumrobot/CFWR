// Test case for issue 93: https://github.com/kelloggm/checker-framework/issues/93

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class ArrayCreationParam {

    public static void m1() {
        int n = 5;
        int[] a = new int[n + 1];
        // Index Checker correctly issues no warning on the lines below
        int j = n;
        if(TestHelper.nondet()) Contract.Assert(j < a.Length);
        int k = n;
        if(TestHelper.nondet()) Contract.Assert(k >= 0 && k < a.Length);
        for (int i = 1; i <= n; i++) {
            int x = a[i];
        }
    }
}
