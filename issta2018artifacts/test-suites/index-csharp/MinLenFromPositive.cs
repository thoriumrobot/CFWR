using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class MinLenFromPositive {

    void test(int x) {
        Contract.Requires(x > 0);
        int[] y = new int[x];
        if(TestHelper.nondet()) Contract.Assert(y.Length >= 1);
        int z = x;
        if(TestHelper.nondet()) Contract.Assert(z >= 1);
        int q = x;
        if(TestHelper.nondet()) Contract.Assert(q > 0);
    }

    /*@SuppressWarnings("index")*/
    void foo(int x) {
        test(x);
    }

    void foo2(int x) {
        // :: error: (argument.type.incompatible)
        test(x);
    }

    void test_lub1(bool flag, int x, int y) {
        Contract.Requires(x > 0);
        Contract.Requires(y >= 6 && y <= 25);
        int z;
        if (flag) {
            z = x;
        } else {
            z = y;
        }
        int q = z;
        if(TestHelper.nondet()) Contract.Assert(q > 0);
        int w = z;
        if(TestHelper.nondet()) Contract.Assert(w >= 1);
    }

    void test_lub2(bool flag, int x, int y) {
        Contract.Requires(x > 0);
        Contract.Requires(y >= -1 && y <= 11);
        int z;
        if (flag) {
            z = x;
        } else {
            z = y;
        }
        // :: error: (assignment.type.incompatible)
        int q = z;
        if(TestHelper.nondet()) Contract.Assert(q > 0);
        int w = z;
        if(TestHelper.nondet()) Contract.Assert(w >= -1);
    }

    int id(int x) {
        Contract.Requires(x > 0);
        Contract.Ensures(Contract.Result<int>() > 0);
        return x;
    }

    void test_id(int param) {
        int x = id(5);
        if(TestHelper.nondet()) Contract.Assert(x > 0);
        int y = id(5);
        if(TestHelper.nondet()) Contract.Assert(y >= 1);

        int[] a = new int[id(100)];

        if(TestHelper.nondet()) Contract.Assert(a.Length >= 1);
        // :: error: (assignment.type.incompatible)
        int[] c = new int[id(100)];
        if(TestHelper.nondet()) Contract.Assert(c.Length >= 10);

        int q = id(10);

        if (param == q) {
            int[] d = new int[param];
            if(TestHelper.nondet()) Contract.Assert(d.Length >= 1);
        }
    }
}
