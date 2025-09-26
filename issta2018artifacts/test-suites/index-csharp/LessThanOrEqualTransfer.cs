using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class LessThanOrEqualTransfer {
    void lte_check(int[] a) {
        if (1 <= a.Length) {
            int[] b = a;
            if(TestHelper.nondet()) Contract.Assert(b.Length >= 1);
        }
    }

    void lte_bad_check(int[] a) {
        if (1 <= a.Length) {
            // :: error: (assignment.type.incompatible)
            int[] b = a;
            if(TestHelper.nondet()) Contract.Assert(b.Length >= 2);
        }
    }
}
