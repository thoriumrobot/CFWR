using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class GreaterThanTransfer {
    void gt_check(int[] a) {
        if (a.Length > 0) {
            int[] b = a;
            if(TestHelper.nondet()) Contract.Assert(b.Length >= 1);
        }
    }

    void gt_bad_check(int[] a) {
        if (a.Length > 0) {
            // :: error: (assignment.type.incompatible)
            int[] b = a;
            if(TestHelper.nondet()) Contract.Assert(b.Length >= 2);
        }
    }
}
