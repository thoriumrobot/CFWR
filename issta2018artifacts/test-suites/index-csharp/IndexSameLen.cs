using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class IndexSameLen {

    public static void bug2() {
        int[] a = {1, 2, 3, 4, 5};
        int[] b = a;
        if(TestHelper.nondet()) Contract.Assert(b.Length == a.Length);

        int i = 2;

        if(TestHelper.nondet()) Contract.Assert(i >= 0 && i < a.Length);
        a[i] = b[i];

        for (int j = 0; j < a.Length; j++) {
            a[j] = b[j];
        }
    }
}
