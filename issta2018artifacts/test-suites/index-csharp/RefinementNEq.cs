using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class RefinementNEq {

    void test_not_equal(int a, int j, int s) {

        // :: error: (assignment.type.incompatible)
        int aa = a;
        if (TestHelper.nondet()) Contract.Assert(aa >= 0);
        if (-1 != a) {
            // :: error: (assignment.type.incompatible)
            int b = a;
            if (TestHelper.nondet()) Contract.Assert(b >= -1);
        } else {
            int c = a;
            if (TestHelper.nondet()) Contract.Assert(c >= -1);
        }

        if (0 != j) {
            // :: error: (assignment.type.incompatible)
            int k = j;
            if (TestHelper.nondet()) Contract.Assert(k >= 0);
        } else {
            int l = j;
            if (TestHelper.nondet()) Contract.Assert(l >= 0);
        }

        if (1 != s) {
            // :: error: (assignment.type.incompatible)
            int t = s;
            if (TestHelper.nondet()) Contract.Assert(t > 0);
        } else {
            int u = s;
            if (TestHelper.nondet()) Contract.Assert(u > 0);
        }
    }
}
// a comment