// Test case for issue #56:
// https://github.com/kelloggm/checker-framework/issues/56

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class EndsWith {

    void testEndsWith(String arg) {
        if (arg.EndsWith("[]", StringComparison.Ordinal)) {
            String arg2 = arg;
            if(TestHelper.nondet()) Contract.Assert(arg.Length >= 2);
        }
    }
}
