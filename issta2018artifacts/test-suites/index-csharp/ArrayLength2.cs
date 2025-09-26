// Test case for issue #14:
// https://github.com/kelloggm/checker-framework/issues/14

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class ArrayLength2 {
    public static void main(String[] args) {
        int N = 8;
        int[] Grid = new int[N];
        if(TestHelper.nondet()) Contract.Assert(Grid.Length >= 8);
        int i = 0;
        if(TestHelper.nondet()) Contract.Assert(i < Grid.Length);
    }
}
