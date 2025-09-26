using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class GreaterThanOrEqualTransfer {
    void gte_check(int[] a) {
        if (a.Length >= 1) {
            int[] b = a;
            if(TestHelper.nondet()) Contract.Assert(b.Length >= 1);
        }
    }

    void gte_bad_check(int[] a) {
        if (a.Length >= 1) {
            // :: error: (assignment.type.incompatible)
            int[] b = a;
            if(TestHelper.nondet()) Contract.Assert(b.Length >= 2);
        }
    }
}
