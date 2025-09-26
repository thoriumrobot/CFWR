using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class Boilerplate {

    void test() {
        // :: error: (assignment.type.incompatible)
        int a = -1;
        if(TestHelper.nondet()) Contract.Assert(a > 0);
    }
}
// a comment
