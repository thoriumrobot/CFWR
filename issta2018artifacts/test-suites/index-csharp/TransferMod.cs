using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;


class TransferMod {

    void test() {
        int aa = -100;
        int a = -1;
        int b = 0;
        int c = 1;
        int d = 2;

        int e = 5 % 3;

        if(TestHelper.nondet()) Contract.Assert(e > 0);
        int f = -100 % 1;
        if(TestHelper.nondet()) Contract.Assert(f >= 0);

        int g = aa % -1;

        if(TestHelper.nondet()) Contract.Assert(g >= 0);
        int h = aa % 1;
        if(TestHelper.nondet()) Contract.Assert(h >= 0);
        int i = d % -1;
        if(TestHelper.nondet()) Contract.Assert(i >= 0);
        int j = d % 1;
        if(TestHelper.nondet()) Contract.Assert(j >= 0);

        int k = d % c;

        if(TestHelper.nondet()) Contract.Assert(k >= 0);
        int l = b % c;
        if(TestHelper.nondet()) Contract.Assert(l >= 0);
        int m = c % d;
        if(TestHelper.nondet()) Contract.Assert(m >= 0);

        int n = c % a;

        if(TestHelper.nondet()) Contract.Assert(n >= 0);
        int o = b % a;
        if(TestHelper.nondet()) Contract.Assert(o >= 0);

        int p = a % a;

        if(TestHelper.nondet()) Contract.Assert(p >= -1);
        int q = a % d;
        if(TestHelper.nondet()) Contract.Assert(q >= -1);
        int r = a % c;
        if(TestHelper.nondet()) Contract.Assert(r >= -1);
    }
}
// a comment
