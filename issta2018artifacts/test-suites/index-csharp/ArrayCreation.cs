using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

// Check that creating an array with the length of another
// makes both @SameLen of each other.

class ArrayCreation {
    void test(int[] a, int[] d) {
        int[] b = new int[a.Length];
        int[] c = b;
        if(TestHelper.nondet()) Contract.Assert(c.Length == a.Length && c.Length == b.Length);
    }
}
