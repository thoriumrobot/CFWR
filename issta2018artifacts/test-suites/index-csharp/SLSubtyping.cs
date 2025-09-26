using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

// This test checks whether the SameLen type system works as expected.

class SLSubtyping {
    int[] f = {1};

    void subtype(int [] a, int[] b) {
        Contract.Requires(a.Length == b.Length);
        int[] c = a;
        if(TestHelper.nondet()) Contract.Assert(c.Length == a.Length && c.Length == b.Length);

        // :: error: (assignment.type.incompatible)
        int[] q = {1, 2};
        if(TestHelper.nondet()) Contract.Assert(q.Length == c.Length);
        int[] d = q;
        if(TestHelper.nondet()) Contract.Assert(d.Length == c.Length);

        // :: error: (assignment.type.incompatible)
        int[] e = a;
        if(TestHelper.nondet()) Contract.Assert(e.Length == f.Length);
    }

    void subtype2(int[] a, int [] b) {
        Contract.Requires(b.Length == a.Length);
        a = b;
        if(TestHelper.nondet()) Contract.Assert(b.Length == a.Length);
        int[] c = b;
        if(TestHelper.nondet()) Contract.Assert(c.Length == b.Length);
        int[] d = f;
        if(TestHelper.nondet()) Contract.Assert(d.Length == f.Length);
    }
}
