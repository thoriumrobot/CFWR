using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;


public class Index132 {
    public static String [] esc_quantify(params String [] vars) {
        Contract.Requires(vars.Length == 1 || vars.Length == 2);
        Contract.Ensures(Contract.Result<String[]>().Length == 3 || Contract.Result<String[]>().Length == 4);
        if (vars.Length == 1) {
            return new String[] {"hello", vars[0], ")"};
        } else {
            return new String[] {"hello", vars[0], vars[1], ")"};
        }
    }
}
