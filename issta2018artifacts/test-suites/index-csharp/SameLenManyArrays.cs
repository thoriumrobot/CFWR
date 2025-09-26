using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class SameLenManyArrays {
    void transfer1(int [] a, int[] b) {
        Contract.Requires(a.Length == b.Length);
        int[] c = new int[a.Length];
        for (int i = 0; i < c.Length; i++) { // i's type is @LTL("c")
            b[i] = 1;
            int[] d = c;
            if(TestHelper.nondet()) Contract.Assert(d.Length == a.Length && d.Length == b.Length && d.Length == c.Length);
        }
    }

    void transfer2(int [] a, int[] b) {
        Contract.Requires(a.Length == b.Length);
        for (int i = 0; i < b.Length; i++) { // i's type is @LTL("b")
            a[i] = 1;
        }
    }

    void transfer3(int [] a, int[] b, int[] c) {
        Contract.Requires(a.Length == b.Length);
        if (a.Length == c.Length) {
            for (int i = 0; i < c.Length; i++) { // i's type is @LTL("c")
                b[i] = 1;
                int[] d = c;
                if(TestHelper.nondet()) Contract.Assert(d.Length == a.Length && d.Length == b.Length && d.Length == c.Length);
            }
        }
    }

    void transfer4(int[] a, int[] b, int[] c) {
        if (b.Length == c.Length) {
            if (a.Length == b.Length) {
                for (int i = 0; i < c.Length; i++) { // i's type is @LTL("c")
                    a[i] = 1;
                    int[] d = c;
                    if(TestHelper.nondet()) Contract.Assert(d.Length == a.Length && d.Length == b.Length && d.Length == c.Length);
                }
            }
        }
    }

    void transfer5(int[] a, int[] b, int[] c, int[] d) {
        if (a.Length == b.Length && b.Length == c.Length) {
            int[] x = a;
            int[] y = x;
            int index = x.Length - 1;
            if (index > 0) {
                f(a[index]);
                f(b[index]);
                f(c[index]);
                f(x[index]);
                f(y[index]);
            }
        }
    }

    [Pure]
    void f(Object o) {}
}
