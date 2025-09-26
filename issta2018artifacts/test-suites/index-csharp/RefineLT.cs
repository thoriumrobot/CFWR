using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class RefineLT {
    int[] arr = {1};

    void testLTL(int test, int a, int a3) {
        Contract.Requires(test < arr.Length);
        Contract.Requires(a < arr.Length);
        Contract.Requires(a3 < arr.Length);
        int b = 2;
        if (b < test) {
            int c = b;
            if(TestHelper.nondet()) Contract.Assert(c < arr.Length);
        }
        // :: error: (assignment.type.incompatible)
        int c1 = b;
        if(TestHelper.nondet()) Contract.Assert(c1 < arr.Length);

        if (b < a3) {
            int potato = 7;
        } else {
            // :: error: (assignment.type.incompatible)
            int d = b;
            if(TestHelper.nondet()) Contract.Assert(d < arr.Length);
        }
    }

    void testLTEL(int test) {
        Contract.Requires(test<arr.Length);
        // :: error: (assignment.type.incompatible)
        int a = int.Parse("1");
        if(TestHelper.nondet()) Contract.Assert(a <= arr.Length);

        // :: error: (assignment.type.incompatible)
        int a3 = int.Parse("3");
        if(TestHelper.nondet()) Contract.Assert(a3 <= arr.Length);

        int b = 2;
        if (b < test) {
            int c = b;
            if(TestHelper.nondet()) Contract.Assert(c <= arr.Length);
        }
        // :: error: (assignment.type.incompatible)
        int c1 = b;
        if(TestHelper.nondet()) Contract.Assert(c1 <= arr.Length);

        if (b < a) {
            int potato = 7;
        } else {
            // :: error: (assignment.type.incompatible)
            int d = b;
            if(TestHelper.nondet()) Contract.Assert(d <= arr.Length);
        }
    }
}
