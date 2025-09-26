using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class SameLenNewArrayWithSameLength {
    public void m1(int[] a) {
        int[] b = new int[a.Length];
        if(TestHelper.nondet()) Contract.Assert(b.Length == a.Length);
    }

    public void m2(int[] a, int [] b) {
        Contract.Requires(b.Length == a.Length);
        int[] c = new int[b.Length];
        if(TestHelper.nondet()) Contract.Assert(c.Length == a.Length && c.Length == b.Length);
    }
}
