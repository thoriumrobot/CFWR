using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class IntroRules {

    void test() {
        int a = 10;
        if(TestHelper.nondet()) Contract.Assert(a > 0);
        int b = 9;
        if(TestHelper.nondet()) Contract.Assert(b >= 0);
        int c = 8;
        if(TestHelper.nondet()) Contract.Assert(c >= -1);
        int d = 7;
        if(TestHelper.nondet()) Contract.Assert(true);

        // :: error: (assignment.type.incompatible)
        int e = 0;
        if(TestHelper.nondet()) Contract.Assert(e > 0);
        // :: error: (assignment.type.incompatible)
        int f = -1;
        if(TestHelper.nondet()) Contract.Assert(f > 0);
        // :: error: (assignment.type.incompatible)
        int g = -6;
        if(TestHelper.nondet()) Contract.Assert(g > 0);

        int h = 0;

        if(TestHelper.nondet()) Contract.Assert(h >= 0);
        int i = 0;
        if(TestHelper.nondet()) Contract.Assert(i >= -1);
        int j = 0;
        if(TestHelper.nondet()) Contract.Assert(true);
        // :: error: (assignment.type.incompatible)
        int k = -1;
        if(TestHelper.nondet()) Contract.Assert(k >= 0);
        // :: error: (assignment.type.incompatible)
        int l = -4;
        if(TestHelper.nondet()) Contract.Assert(l >= 0);

        int m = -1;

        if(TestHelper.nondet()) Contract.Assert(m >= -1);
        int n = -1;
        if(TestHelper.nondet()) Contract.Assert(true);
        // :: error: (assignment.type.incompatible)
        int o = -9;
        if(TestHelper.nondet()) Contract.Assert(o >= -1);
    }
}
// a comment
