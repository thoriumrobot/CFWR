using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class IntroAnd {
    void test() {
        int a = 1 & 0;
        if(TestHelper.nondet()) Contract.Assert(a >= 0);
        int b = a & 5;
        if(TestHelper.nondet()) Contract.Assert(b >= 0);

        // :: error: (assignment.type.incompatible)
        int c = a & b;
        if(TestHelper.nondet()) Contract.Assert(c > 0);
        int d = a & b;
        if(TestHelper.nondet()) Contract.Assert(d >= 0);
        int e = b & a;
        if(TestHelper.nondet()) Contract.Assert(e >= 0);
    }

    void test_ubc_and(
            int i, int[] a, int j, int k, int m) {
        Contract.Requires(i >= 0 && i<a.Length);
        Contract.Requires(j<a.Length);
        Contract.Requires(m >= 0);
        int x = a[i & k];
        int x1 = a[k & i];
        // :: error: (array.access.unsafe.low) :: error: (array.access.unsafe.high.range)
        int y = a[j & k];
        if (j > -1) {
            int z = a[j & k];
        }
        // :: error: (array.access.unsafe.high.range)
        int w = a[m & k];
        if (m < a.Length) {
            int u = a[m & k];
        }
    }

    void two_arrays(int[] a, int[] b, int i, int j) {
        Contract.Requires(i >= 0 && i<a.Length);
        Contract.Requires(j >= 0 && j<b.Length);
        int l = a[i & j];
        l = b[i & j];
    }

    void test_pos(int x, int y) {
        Contract.Requires(x > 0);
        Contract.Requires(y > 0);
        // :: error: (assignment.type.incompatible)
        int z = x & y;
        if(TestHelper.nondet()) Contract.Assert(z > 0);
    }
}
