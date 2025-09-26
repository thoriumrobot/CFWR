// Test case for issue #66:
// https://github.com/kelloggm/checker-framework/issues/66

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class ArrayConstructionPositiveLength {

    public void makeArray(int max_values) {
        Contract.Requires(max_values > 0);
        String[] a = new String[max_values];
        if(TestHelper.nondet()) Contract.Assert(a.Length >= 1);
    }
}
