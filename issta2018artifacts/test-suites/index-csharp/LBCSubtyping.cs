using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class LBCSubtyping {

    void foo() {

        int i = -1;

        if(TestHelper.nondet()) Contract.Assert(i >= -1);

        int j = i;

        if(TestHelper.nondet()) Contract.Assert(true);

        int k = -4;

        // not this one though
        // :: error: (assignment.type.incompatible)
        int l = k;
        if(TestHelper.nondet()) Contract.Assert(l >= -1);

        int n = 0;

        if(TestHelper.nondet()) Contract.Assert(n >= 0);

        int a = 1;

        if(TestHelper.nondet()) Contract.Assert(a > 0);

        // check that everything is aboveboard
        j = a;
        if(TestHelper.nondet()) Contract.Assert(true);
        j = n;
        if(TestHelper.nondet()) Contract.Assert(true);
        l = n;
        if(TestHelper.nondet()) Contract.Assert(l >= -1);
        n = a;
        if(TestHelper.nondet()) Contract.Assert(n >= 0);

        // error cases

        // :: error: (assignment.type.incompatible)
        int p = i;
        if(TestHelper.nondet()) Contract.Assert(p >= 0);
        // :: error: (assignment.type.incompatible)
        int b = i;
        if(TestHelper.nondet()) Contract.Assert(b > 0);

        // :: error: (assignment.type.incompatible)
        int r = k;
        if(TestHelper.nondet()) Contract.Assert(r >= 0);
        // :: error: (assignment.type.incompatible)
        int c = k;
        if(TestHelper.nondet()) Contract.Assert(c > 0);

        // :: error: (assignment.type.incompatible)
        int d = r;
        if(TestHelper.nondet()) Contract.Assert(d > 0);
    }
}
// a comment
