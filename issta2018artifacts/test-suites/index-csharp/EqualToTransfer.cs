using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class EqualToTransfer {
    void eq_check(int[] a) {
        if (1 == a.Length) {
            int[] b = a;
            if(TestHelper.nondet()) Contract.Assert(b.Length >= 1);
        }
        if (a.Length == 1) {
            int[] b = a;
            if(TestHelper.nondet()) Contract.Assert(b.Length >= 1);
        }
    }

    void eq_bad_check(int[] a) {
        if (1 == a.Length) {
            // :: error: (assignment.type.incompatible)
            int[] b = a;
            if(TestHelper.nondet()) Contract.Assert(b.Length >= 2);
        }
    }
}
