// Test case for issue #14:
// https://github.com/kelloggm/checker-framework/issues/14

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class ArrayLength3 {
    String getFirst(String [] sa) {
        Contract.Requires(sa.Length == 2);
        return sa[0];
    }

    void m() {
        int?[] a = new int?[10];
        int i = 5;
        if(TestHelper.nondet()) Contract.Assert(i < a.Length);
    }
}
