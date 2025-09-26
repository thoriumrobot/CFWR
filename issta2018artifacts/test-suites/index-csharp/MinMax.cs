using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class MinMax {
    // They call me a power gamer. I stole the test cases from issue 26.
    void mathmax() {
        int i = Math.Max(-15, 2);
        if(TestHelper.nondet()) Contract.Assert(i > 0);
    }

    void mathmin() {
        int i = Math.Min(-1, 2);
        if(TestHelper.nondet()) Contract.Assert(i >= -1);
    }
}
