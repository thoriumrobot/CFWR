// Test case for issue #168: https://github.com/kelloggm/checker-framework/issues/168

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class EndsWith2 {

    public static String invertBrackets(String classname) {

        // Get the array depth (if any)
        int array_depth = 0;
        String brackets = "";
        while (classname.EndsWith("[]", StringComparison.Ordinal)) {
            brackets = brackets + classname.Substring(classname.Length - 2);
            classname = classname.Substring(0, classname.Length - 2);
            array_depth++;
        }
        return brackets + classname;
    }
}
