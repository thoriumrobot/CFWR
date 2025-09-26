using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

// From plume-lib's OrderedPairIterator class.
// If quals are configured incorrectly, there will be an
// incompatible assignment error; this ensures that Void
// is given the Positive type.

public class IteratorVoid<T> where T:class{
    protected T next1;
    protected IEnumerator<T> itor1;

    private void setnext1() {
        next1 = itor1.MoveNext() ? itor1.Current : null;
    }
}
