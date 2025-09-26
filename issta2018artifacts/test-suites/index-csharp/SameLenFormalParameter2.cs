using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class SameLenFormalParameter2 {

    void lib(Object [] valsArg, int [] modsArg) {
        Contract.Requires(valsArg.Length == valsArg.Length && valsArg.Length == modsArg.Length && modsArg.Length == modsArg.Length);
    }

    void client(Object[] myvals, int[] mymods) {
        // :: error: (argument.type.incompatible)
        lib(myvals, mymods);
    }
}
