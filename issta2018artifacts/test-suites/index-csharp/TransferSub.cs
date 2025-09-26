using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class TransferSub {

    void test() {
        // zero, one, and two
        int a = 1;

        int b = a - 1;

        if(TestHelper.nondet()) Contract.Assert(b >= 0);
        // :: error: (assignment.type.incompatible)
        int c = a - 1;
        if(TestHelper.nondet()) Contract.Assert(c > 0);
        int d = a - 2;
        if(TestHelper.nondet()) Contract.Assert(d >= -1);

        // :: error: (assignment.type.incompatible)
        int e = a - 2;
        if(TestHelper.nondet()) Contract.Assert(e >= 0);

        int f = b - 1;

        if(TestHelper.nondet()) Contract.Assert(f >= -1);
        // :: error: (assignment.type.incompatible)
        int g = b - 1;
        if(TestHelper.nondet()) Contract.Assert(g >= 0);

        // :: error: (assignment.type.incompatible)
        int h = f - 1;
        if(TestHelper.nondet()) Contract.Assert(h >= -1);

        int i = f - 0;

        if(TestHelper.nondet()) Contract.Assert(i >= -1);
        int j = b - 0;
        if(TestHelper.nondet()) Contract.Assert(j >= 0);
        int k = a - 0;
        if(TestHelper.nondet()) Contract.Assert(k > 0);

        // :: error: (assignment.type.incompatible)
        int l = j - 0;
        if(TestHelper.nondet()) Contract.Assert(l > 0);
        // :: error: (assignment.type.incompatible)
        int m = i - 0;
        if(TestHelper.nondet()) Contract.Assert(m >= 0);

        // :: error: (assignment.type.incompatible)
        int n = a - k;
        if(TestHelper.nondet()) Contract.Assert(n > 0);
        // this would be an error if the values of b and j (both zero) weren't known at compile time
        int o = b - j;
        if(TestHelper.nondet()) Contract.Assert(o >= 0);
        /* i and d both have compile time value -1, so this is legal.
        The general case of GTEN1 - GTEN1 is not, though. */
        int p = i - d;
        if(TestHelper.nondet()) Contract.Assert(p >= -1);

        // decrements

        // :: error: (compound.assignment.type.incompatible) :: error: (assignment.type.incompatible)
        int q = --k; // k = 0
        if(TestHelper.nondet()) Contract.Assert(q > 0);
        if(TestHelper.nondet()) Contract.Assert(k > 0);

        // :: error: (compound.assignment.type.incompatible)
        int r = k--; // after this k = -1
        if(TestHelper.nondet()) Contract.Assert(r >= 0);
        if(TestHelper.nondet()) Contract.Assert(k > 0);

        int k1 = 0;
        int s = k1--;
        if(TestHelper.nondet()) Contract.Assert(s >= 0);

        // :: error: (assignment.type.incompatible)
        int s1 = k1;
        if(TestHelper.nondet()) Contract.Assert(s1 >= 0);

        // transferred to SimpleTransferSub.java
        // this section is failing due to CF bug
        // int k2 = 0;
        // // :: error: (assignment.type.incompatible)
        // @Positive int s2 = k2--;

        k1 = 1;
        int t = --k1;
        if(TestHelper.nondet()) Contract.Assert(t >= 0);

        k1 = 1;
        // :: error: (assignment.type.incompatible)
        int t1 = --k1;
        if(TestHelper.nondet()) Contract.Assert(t1 > 0);

        int u1 = -1;
        int x = u1--;
        if(TestHelper.nondet()) Contract.Assert(x >= -1);
        // :: error: (assignment.type.incompatible)
        int x1 = u1;
        if(TestHelper.nondet()) Contract.Assert(x1 >= -1);
    }
}
// a comment
