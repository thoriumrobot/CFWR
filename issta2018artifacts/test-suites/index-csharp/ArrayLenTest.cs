using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class ArrayLenTest {
    public static String esc_quantify(params String[] vars) {
        Contract.Requires(vars.Length == 1 || vars.Length == 2);
        if (vars.Length == 1) {
            return vars[0];
        } else {
            int i = vars.Length;
            if(TestHelper.nondet()) Contract.Assert(i == 2);
            String[] a = vars;
            if(TestHelper.nondet()) Contract.Assert(a.Length == 2);
            return vars[0] + vars[1];
        }
    }
}
