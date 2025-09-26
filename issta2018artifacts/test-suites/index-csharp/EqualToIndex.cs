using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class EqualToIndex {
    public static int[] a = {0};

    public static void equalToUpper( int m, int r) {
        Contract.Requires(m<a.Length);
        Contract.Requires(r<=a.Length);
        if (r == m) {
            int j = r;
            if(TestHelper.nondet()) Contract.Assert(j < a.Length);
        }
    }
}
