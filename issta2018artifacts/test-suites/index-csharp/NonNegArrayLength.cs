using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class NonNegArrayLength {

    public static void NonNegArrayLength_(int [] arr) {
        Contract.Requires(arr.Length >= 4);
        int i = arr.Length - 2;
        if(TestHelper.nondet()) Contract.Assert(i > 0);
    }
}
