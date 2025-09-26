// Test case for issue 97: https://github.com/kelloggm/checker-framework/issues/97

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class Offset97 {
    public static void m2() {
        int[] a = {1, 2, 3, 4, 5};
        int i = 4;
        if(TestHelper.nondet()) Contract.Assert(i >= 0 && i < a.Length);
        int j = 4;
        if(TestHelper.nondet()) Contract.Assert(j >= 0 && j < a.Length);
        if (j < a.Length - i) {
            int k = i + j;
            if(TestHelper.nondet()) Contract.Assert(k >= 0 && k < a.Length);
        }
    }
}
