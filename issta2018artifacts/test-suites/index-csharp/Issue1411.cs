// Test case for Issue 1411:
// https://github.com/typetools/checker-framework/issues/1411

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

interface IGeneric<V> {
    [Pure]
    V get();
}

interface IConcrete : IGeneric<char[]> {}

public class Issue1411 {
    static void m(IConcrete ic) {
        char[] val = ic.get();
    }
}
