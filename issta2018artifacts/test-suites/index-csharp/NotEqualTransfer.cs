using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class NotEqualTransfer {
    void neq_check(int[] a) {
        if (1 != a.Length) {
            int x = 1; // do nothing.
        } else {
            int[] b = a;
            if(TestHelper.nondet()) Contract.Assert(b.Length >= 1);
        }
    }

    void neq_bad_check(int[] a) {
        if (1 != a.Length) {
            int x = 1; // do nothing.
        } else {
            // :: error: (assignment.type.incompatible)
            int[] b = a;
            if(TestHelper.nondet()) Contract.Assert(b.Length >= 2);
        }
    }

    void neq_zero_special_case(int[] a) {
        if (a.Length != 0) {
            int[] b = a;
            if(TestHelper.nondet()) Contract.Assert(b.Length >= 1);
        }
    }
}
