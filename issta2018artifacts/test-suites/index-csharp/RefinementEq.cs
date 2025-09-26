using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class RefinementEq {

    void test_equal(int a, int j, int s) {

        if (-1 == a) {
            int b = a;
            if(TestHelper.nondet()) Contract.Assert(b >= -1);
        } else {
            // :: error: (assignment.type.incompatible)
            int c = a;
            if(TestHelper.nondet()) Contract.Assert(c >= -1);
        }

        if (0 == j) {
            int k = j;
            if(TestHelper.nondet()) Contract.Assert(k >= 0);
        } else {
            // :: error: (assignment.type.incompatible)
            int l = j;
            if(TestHelper.nondet()) Contract.Assert(l >= 0);
        }

        if (1 == s) {
            int t = s;
            if(TestHelper.nondet()) Contract.Assert(t > 0);
        } else {
            // :: error: (assignment.type.incompatible)
            int u = s;
            if(TestHelper.nondet()) Contract.Assert(u > 0);
        }
    }
}
// a comment