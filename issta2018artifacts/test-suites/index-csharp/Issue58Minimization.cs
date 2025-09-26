using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class Issue58Minimization {

    void test(int x) {
        Contract.Requires(x >= -1);
        int z;
        if ((z = x) != -1) {
            int y = z;
            if(TestHelper.nondet()) Contract.Assert(y >= 0);
        }
        if ((z = x) != 1) {
            // :: error: (assignment.type.incompatible)
            int y = z;
            if(TestHelper.nondet()) Contract.Assert(y >= 0);
        }
    }

    void test2(int x) {
        Contract.Requires(x >= 0);
        int z;
        if ((z = x) != 0) {
            int y = z;
            if(TestHelper.nondet()) Contract.Assert(y > 0);
        }
        if ((z = x) == 0) {
            // do nothing
            int y = 5;
        } else {
            int y = x;
            if(TestHelper.nondet()) Contract.Assert(y > 0);
        }
    }

    void ubc_test(int[] a, int x) {
        Contract.Requires(x <= a.Length);
        int z;
        if ((z = x) != a.Length) {
            int y = z;
            if(TestHelper.nondet()) Contract.Assert(y < a.Length);
        }
    }

    void samelen_test(int[] a, int[] c) {
        int[] b;
        if ((b = a) == c) {
            int[] d = b;
            if(TestHelper.nondet()) Contract.Assert(d.Length == a.Length && d.Length == b.Length && d.Length == c.Length);
        }
    }

    void minlen_test(int[] a, int [] c) {
        Contract.Requires(c.Length >= 1);
        int[] b;
        if ((b = a) == c) {
            int[] d = b;
            if(TestHelper.nondet()) Contract.Assert(d.Length >= 1);
        }
    }

    void minlen_test2(int[] a, int x) {
        int one = 1;
        if ((x = one) == a.Length) {
            int[] b = a;
            if(TestHelper.nondet()) Contract.Assert(b.Length >= 1);
        }
    }
}
