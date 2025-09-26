using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class TransferTimes {

    void test() {
        int a = 1;
        int b = a * 1;
        if(TestHelper.nondet()) Contract.Assert(b > 0);
        int c = 1 * a;
        if(TestHelper.nondet()) Contract.Assert(c > 0);
        int d = 0 * a;
        if(TestHelper.nondet()) Contract.Assert(d >= 0);
        // :: error: (assignment.type.incompatible)
        int e = -1 * a;
        if(TestHelper.nondet()) Contract.Assert(e >= 0);

        int g = -1;
        int h = g * 0;
        if(TestHelper.nondet()) Contract.Assert(h >= 0);
        // :: error: (assignment.type.incompatible)
        int i = g * 0;
        if(TestHelper.nondet()) Contract.Assert(i > 0);
        // :: error: (assignment.type.incompatible)
        int j = g * a;
        if(TestHelper.nondet()) Contract.Assert(j > 0);

        int k = 0;
        int l = 1;
        int m = a * l;
        if(TestHelper.nondet()) Contract.Assert(m > 0);
        int n = k * l;
        if(TestHelper.nondet()) Contract.Assert(n >= 0);
        int o = k * k;
        if(TestHelper.nondet()) Contract.Assert(o >= 0);
    }
}
// a comment