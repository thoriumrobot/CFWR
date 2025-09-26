using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class CheckNotNull2<T> where T:class {
    T checkNotNull(T @ref) {
        return @ref;
    }

    void test(T @ref) {
        checkNotNull(@ref);
    }
}
