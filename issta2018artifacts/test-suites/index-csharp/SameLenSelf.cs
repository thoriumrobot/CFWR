// Test case for issue 146: https://github.com/kelloggm/checker-framework/issues/146

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class SameLenSelf {
    int [] field = new int[10];
    int [] field2 = new int[10];
    int[] field3;
    SameLenSelf() {
        field3 = field2;
    }

    [ContractInvariantMethod]
    private void Invariant()
    {
        Contract.Invariant(field.Length == this.field.Length);
        Contract.Invariant(field2.Length == field2.Length);
        Contract.Invariant(field3.Length == field3.Length);
    }

    void foo(int[] b) {
        int[] a = b;
        if(TestHelper.nondet()) Contract.Assert(a.Length == a.Length);
        int[] c = new int[10];
        if(TestHelper.nondet()) Contract.Assert(c.Length == c.Length);
    }
}
