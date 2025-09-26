using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class TransferAdd {

    void test() {

        // adding zero and one and two

        int a = -1;

        int a1 = a + 2;
        if(TestHelper.nondet()) Contract.Assert(a1 > 0);

        int b = a + 1;
        if(TestHelper.nondet()) Contract.Assert(b >= 0);
        int c = 1 + a;
        if(TestHelper.nondet()) Contract.Assert(c >= 0);

        int d = a + 0;
        if(TestHelper.nondet()) Contract.Assert(d >= -1);
        int e = 0 + a;
        if(TestHelper.nondet()) Contract.Assert(e >= -1);

        // :: error: (assignment.type.incompatible)
        int f = a + 1;
        if(TestHelper.nondet()) Contract.Assert(f > 0);

        int g = b + 0;

        if(TestHelper.nondet()) Contract.Assert(g >= 0);

        int h = b + 1;

        if(TestHelper.nondet()) Contract.Assert(h > 0);

        int i = h + 1;

        if(TestHelper.nondet()) Contract.Assert(i > 0);
        int j = h + 0;
        if(TestHelper.nondet()) Contract.Assert(j > 0);

        // adding values

        int k = i + j;

        if(TestHelper.nondet()) Contract.Assert(k > 0);
        // :: error: (assignment.type.incompatible)
        int l = b + c;
        if(TestHelper.nondet()) Contract.Assert(l > 0);
        // :: error: (assignment.type.incompatible)
        int m = d + c;
        if(TestHelper.nondet()) Contract.Assert(m > 0);
        // :: error: (assignment.type.incompatible)
        int n = d + e;
        if(TestHelper.nondet()) Contract.Assert(n > 0);

        int o = h + g;

        if(TestHelper.nondet()) Contract.Assert(o > 0);
        // :: error: (assignment.type.incompatible)
        int p = h + d;
        if(TestHelper.nondet()) Contract.Assert(p > 0);

        int q = b + c;

        if(TestHelper.nondet()) Contract.Assert(q >= 0);
        // :: error: (assignment.type.incompatible)
        int r = q + d;
        if(TestHelper.nondet()) Contract.Assert(r >= 0);

        int s = k + d;

        if(TestHelper.nondet()) Contract.Assert(s >= 0);
        int t = s + d;
        if(TestHelper.nondet()) Contract.Assert(t >= -1);

        // increments

        // :: error: (assignment.type.incompatible)
        int u = b++;
        if(TestHelper.nondet()) Contract.Assert(u > 0);
        if(TestHelper.nondet()) Contract.Assert(b >= 0);

        int u1 = b;

        if(TestHelper.nondet()) Contract.Assert(u1 > 0);

        int v = ++c;

        if(TestHelper.nondet()) Contract.Assert(v > 0);
        if(TestHelper.nondet()) Contract.Assert(c >= 0);

        int v1 = c;

        if(TestHelper.nondet()) Contract.Assert(v1 > 0);

        int n1p1 = -1, n1p2 = -1;

        int w = ++n1p1;

        if(TestHelper.nondet()) Contract.Assert(w >= 0);

        int w1 = n1p1;

        if(TestHelper.nondet()) Contract.Assert(w1 >= 0);

        // :: error: (assignment.type.incompatible)
        int w2 = n1p1;
        if(TestHelper.nondet()) Contract.Assert(w2 > 0);
        // :: error: (assignment.type.incompatible)
        int w3 = n1p1++;
        if(TestHelper.nondet()) Contract.Assert(w3 > 0);

        // :: error: (assignment.type.incompatible)
        int x = n1p2++;
        if(TestHelper.nondet()) Contract.Assert(x >= 0);

        int x1 = n1p2;

        if(TestHelper.nondet()) Contract.Assert(x1 >= 0);

        // :: error: (assignment.type.incompatible)
        int y = ++d;
        if(TestHelper.nondet()) Contract.Assert(y > 0);
        if(TestHelper.nondet()) Contract.Assert(d >= -1);
        // :: error: (assignment.type.incompatible)
        int z = e++;
        if(TestHelper.nondet()) Contract.Assert(z > 0);
    }
}
// a comment
