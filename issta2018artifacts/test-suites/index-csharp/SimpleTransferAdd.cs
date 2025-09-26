using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class SimpleTransferAdd {
    void test() {
        int bs = -1;
        // :: error: (assignment.type.incompatible)
        int es = bs;
        if(TestHelper.nondet()) Contract.Assert(es >= 0);

        // @NonNegative int ds = 2 + bs;
        int ds = 0;
        // :: error: (assignment.type.incompatible)
        int cs = ds++;
        if(TestHelper.nondet()) Contract.Assert(cs > 0);
        int fs = ds;
        if(TestHelper.nondet()) Contract.Assert(fs > 0);
    }
}
// a comment
