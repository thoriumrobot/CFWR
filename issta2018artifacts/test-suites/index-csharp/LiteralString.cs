// Test case for issue #55:
// https://github.com/kelloggm/checker-framework/issues/55

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class LiteralString {

    private static readonly String[] finalField = {"This", "is", "an", "array"};

    void testLiteralString() {
        String s = "This string is long enough";
        if(TestHelper.nondet()) Contract.Assert(s.Length >= 10);
    }

    void testLiteralArray() {
        String[] a = new String[] {"This", "array", "is", "long", "enough"};
        if(TestHelper.nondet()) Contract.Assert(a.Length >= 2);
        String[] b = finalField;
        if(TestHelper.nondet()) Contract.Assert(b.Length >= 2);
        int i = 0;
        if(TestHelper.nondet()) Contract.Assert(i >= 0 && i < finalField.Length);
    }
}
