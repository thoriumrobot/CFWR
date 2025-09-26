using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class RefineEq {
    int[] arr = {1};

    void testLTL(int test) {
        Contract.Requires(test < arr.Length);
        // :: error: (assignment.type.incompatible)
        int a = int.Parse("1");
        if(TestHelper.nondet()) Contract.Assert(a < arr.Length);

        int b = 1;
        if (test == b) {
            int c = b;
            if(TestHelper.nondet()) Contract.Assert(c < arr.Length);

        } else {
            // :: error: (assignment.type.incompatible)
            int e = b;
            if(TestHelper.nondet()) Contract.Assert(e < arr.Length);
        }
        // :: error: (assignment.type.incompatible)
        int d = b;
        if(TestHelper.nondet()) Contract.Assert(d < arr.Length);
    }

    void testLTEL(int test) {
        Contract.Requires(test <= arr.Length);
        // :: error: (assignment.type.incompatible)
        int a = int.Parse("1");
        if(TestHelper.nondet()) Contract.Assert(a <= arr.Length);

        int b = 1;
        if (test == b) {
            int c = b;
            if(TestHelper.nondet()) Contract.Assert(c <= arr.Length);

            // :: error: (assignment.type.incompatible)
            int g = b;
            if(TestHelper.nondet()) Contract.Assert(g < arr.Length);
        } else {
            // :: error: (assignment.type.incompatible)
            int e = b;
            if(TestHelper.nondet()) Contract.Assert(e <= arr.Length);
        }
        // :: error: (assignment.type.incompatible)
        int d = b;
        if(TestHelper.nondet()) Contract.Assert(d <= arr.Length);
    }
}
