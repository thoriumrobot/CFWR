using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;



public class TransferDivide {

    void test() {
        int a = -1;
        int b = 0;
        int c = 1;
        int d = 2;

        /** literals */
        int e = -1 / -1;
        if(TestHelper.nondet()) Contract.Assert(e > 0);

        /** 0 / * -> NN */
        int f = 0 / a;
        if(TestHelper.nondet()) Contract.Assert(f >= 0);
        int g = 0 / d;
        if(TestHelper.nondet()) Contract.Assert(g >= 0);

        /** * / 1 -> * */
        int h = a / 1;
        if(TestHelper.nondet()) Contract.Assert(h >= -1);
        int i = b / 1;
        if(TestHelper.nondet()) Contract.Assert(i >= 0);
        int j = c / 1;
        if(TestHelper.nondet()) Contract.Assert(j > 0);
        int k = d / 1;
        if(TestHelper.nondet()) Contract.Assert(k > 0);

        /** pos / pos -> nn */
        int l = d / c;
        if(TestHelper.nondet()) Contract.Assert(l >= 0);
        int m = c / d;
        if(TestHelper.nondet()) Contract.Assert(m >= 0);
        // :: error: (assignment.type.incompatible)
        int n = c / d;
        if(TestHelper.nondet()) Contract.Assert(n > 0);

        /** nn / pos -> nn */
        int o = b / c;
        if(TestHelper.nondet()) Contract.Assert(o >= 0);
        // :: error: (assignment.type.incompatible)
        int p = b / d;
        if(TestHelper.nondet()) Contract.Assert(p > 0);

        /** pos / nn -> nn */
        int q = d / l;
        if(TestHelper.nondet()) Contract.Assert(q >= 0);
        // :: error: (assignment.type.incompatible)
        int r = c / l;
        if(TestHelper.nondet()) Contract.Assert(r > 0);

        /** nn / nn -> nn */
        int s = b / q;
        if(TestHelper.nondet()) Contract.Assert(s >= 0);
        // :: error: (assignment.type.incompatible)
        int t = b / q;
        if(TestHelper.nondet()) Contract.Assert(t > 0);

        /** n1p / pos -> n1p */
        int u = a / d;
        if(TestHelper.nondet()) Contract.Assert(u >= -1);
        int v = a / c;
        if(TestHelper.nondet()) Contract.Assert(v >= -1);
        // :: error: (assignment.type.incompatible)
        int w = a / c;
        if(TestHelper.nondet()) Contract.Assert(w >= 0);

        /** n1p / nn -> n1p */
        int x = a / l;
        if(TestHelper.nondet()) Contract.Assert(x >= -1);
    }

    void testDivideByTwo(int x) {
        Contract.Requires(x >= 0);
        int y = x / 2;
        if(TestHelper.nondet()) Contract.Assert(y >= 0);
    }
}
// a comment
