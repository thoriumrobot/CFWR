using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class SameLenLUBStrangeness {
    void test(int[] a, bool cond) {
        int[] b = null;
        if (cond) {
            b = a;
        }
        // :: error: (assignment.type.incompatible)
        int[] c = a;
        if(TestHelper.nondet()) Contract.Assert(c.Length == a.Length && c.Length == b.Length);
    }
}
