using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class ArrayLengthEquality {
    void test(int[] a, int[] b) {
        if (a.Length == b.Length) {
            int[] c = a;
            if(TestHelper.nondet()) Contract.Assert(c.Length == a.Length && c.Length == b.Length);
            int[] d = b;
            if(TestHelper.nondet()) Contract.Assert(d.Length == a.Length && d.Length == b.Length);
        }
        if (a.Length != b.Length) {
            // Do nothing.
            int x = 0;
        } else {
            int[] e = a;
            if(TestHelper.nondet()) Contract.Assert(e.Length == a.Length && e.Length == b.Length);
            int[] f = b;
            if(TestHelper.nondet()) Contract.Assert(f.Length == a.Length && f.Length == b.Length);
        }
    }
}
