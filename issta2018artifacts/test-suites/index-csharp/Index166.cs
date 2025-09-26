// Test case for Issue 166:
// https://github.com/kelloggm/checker-framework/issues/166

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class Index166 {

    public void testMethodInvocation() {
        requiresIndex("012345", 5);
        // :: error: (argument.type.incompatible)
        requiresIndex("012345", 6);
    }

    public void requiresIndex(String str, int index) {
        Contract.Requires(index >= 0 && index < str.Length);
    }
}
