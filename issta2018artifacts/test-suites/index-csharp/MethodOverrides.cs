// This class should not issues any errors from the value checker.
// The index checker should issue the errors instead.

// There is a copy of this test at checker/tests/value-index-interaction/MethodOverrides.java,
// which does not include expected failures.

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class MethodOverrides {
    public virtual int read() {
        Contract.Ensures(Contract.Result<int>() >= -1);
        return -1;
    }
}

class MethodOverrides2 : MethodOverrides {
    // :: error: (override.return.invalid)
    public override int read() {
        return -1;
    }
}
