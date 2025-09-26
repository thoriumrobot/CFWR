using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class TypeArrayLengthWithSameLen {
    void test(int [] a, int [] b, int[] c) {
        Contract.Requires(a.Length == b.Length);
        Contract.Requires(b.Length == a.Length);
        if (a.Length == c.Length) {
            int x = b.Length;
            if(TestHelper.nondet()) Contract.Assert(x <= a.Length && x <= b.Length && x <= c.Length);
        }
    }
}
