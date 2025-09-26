using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;


class SameLenEqualsRefinement {
    void transfer3(int [] a, int[] b, int[] c) {
        Contract.Requires(a.Length == b.Length);
        if (a == c) {
            for (int i = 0; i < c.Length; i++) { // i's type is @LTL("c")
                b[i] = 1;
                int[] d = c;
                if(TestHelper.nondet()) Contract.Assert(d.Length == a.Length && d.Length == b.Length && d.Length == c.Length);
            }
        }
    }

    void transfer4(int[] a, int[] b, int[] c) {
        if (b == c) {
            if (a == b) {
                for (int i = 0; i < c.Length; i++) { // i's type is @LTL("c")
                    a[i] = 1;
                    int[] d = c;
                    if(TestHelper.nondet()) Contract.Assert(d.Length == a.Length && d.Length == b.Length && d.Length == c.Length);
                }
            }
        }
    }

    void transfer5(int[] a, int[] b, int[] c, int[] d) {
        if (a == b && b == c) {
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
