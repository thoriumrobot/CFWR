using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class LessThanTransfer {
    void lt_check(int[] a) {
        if (0 < a.Length) {
            int[] b = a;
            if(TestHelper.nondet()) Contract.Assert(b.Length >= 1);
        }
    }

    void lt_bad_check(int[] a) {
        if (0 < a.Length) {
            // :: error: (assignment.type.incompatible)
            int[] b = a;
            if(TestHelper.nondet()) Contract.Assert(b.Length >= 2);
        }
    }
}
