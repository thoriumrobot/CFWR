using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class SimpleTransferSub {
    void test() {
        // shows a bug in the checker framework. I don't think we can get around this bit...
        int bs = 0;
        // :: error: (assignment.type.incompatible)
        int ds = bs--;
        if(TestHelper.nondet()) Contract.Assert(ds > 0);
    }
}
// a comment
