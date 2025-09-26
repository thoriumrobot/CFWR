// Test case for issue 98: https://github.com/kelloggm/checker-framework/issues/98

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class SubtractingNonNegatives {
    public static void m4(int[] a, int i, int j) {
        Contract.Requires(i >= 0 && i<a.Length);
        Contract.Requires(j >= 0 && j<a.Length);
        int k = i;
        if (k >= j) {
            int y = k;
            if(TestHelper.nondet()) Contract.Assert(y >= 0 && y < a.Length);
        }
        for (k = i; k >= j; k -= j) {
            int x = k;
            if(TestHelper.nondet()) Contract.Assert(x >= 0 && x < a.Length);
        }
    }

    /*@SuppressWarnings("lowerbound")*/
    void test(int[] a, int y) {
        Contract.Requires(y > 0);
        int x = a.Length - 1;
        if(TestHelper.nondet()) Contract.Assert(x < a.Length);
        int z = x - y;
        if(TestHelper.nondet()) Contract.Assert(z + 0 <a.Length && z + y < a.Length);
        a[z + y] = 0;
    }

    /*@SuppressWarnings("lowerbound")*/
    void test2(int[] a, int y) {
        Contract.Requires(y > 0);
        int x = a.Length - 1;
        if(TestHelper.nondet()) Contract.Assert(x < a.Length);
        int z = x - y;
        a[z + y] = 0;
    }
}
