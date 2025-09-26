using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class CheckNotNull1 {
    T checkNotNull<T> (T @ref) where T : class
    {
        return @ref;
    }

    void test<S>(S @ref) where S : class{
        checkNotNull(@ref);
    }
}
