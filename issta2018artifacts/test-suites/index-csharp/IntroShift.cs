using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class IntroShift {
    void test() {
        int a = 1 >> 1;
        if(TestHelper.nondet()) Contract.Assert(a >= 0);
        // :: error: (assignment.type.incompatible)
        int b = -1 >> 0;
        if(TestHelper.nondet()) Contract.Assert(b >= 0);
    }
}